import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from collections import OrderedDict
import json
import sys
import hydra
from omegaconf import OmegaConf, open_dict

import __init__

from models import so3conv as M
import vgtk.so3conv.functional as L

def bp():
    import pdb;pdb.set_trace()

class MocapNet(nn.Module):
    def __init__(self, params, config=None):
        super(MocapNet, self).__init__()

        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))

        self.na_in = params['na']
        self.invariance = True
        self.config = config
        self.t_method_type = 2 #config.t_method_type

        # if 'ycb' in config.task and config.instance is None:
        #     self.num_heads = config.DATASET.num_classes
        #     self.classifier = nn.Linear(params['outblock']['mlp'][-1], self.num_heads)
        # else:
        self.num_heads = 24*64
        self.classifier = None
        # per anchors R, T estimation
        # if config.t_method_type == -1:    # 0.845, R_i * delta_T
        #     self.outblockR = M.SO3OutBlockR(params['outblock'], norm=1, pooling_method=config.model.pooling_method, pred_t=config.pred_t, feat_mode_num=self.na_in)
        # elif config.t_method_type == 0:   # 0.847, R_i0 * delta_T
        #     self.outblockRT = M.SO3OutBlockR(params['outblock'], norm=1, pooling_method=config.model.pooling_method,
        #                                      pred_t=config.pred_t, feat_mode_num=self.na_in, num_heads=self.num_heads)
        # elif config.t_method_type == 1: # 0.8472,R_i0 * (xyz + Scalar*delta_T)_mean, current fastest
        #     self.outblockRT = M.SO3OutBlockRT(params['outblock'], norm=1, pooling_method=config.model.pooling_method,
        #                                       global_scalar=True, use_anchors=False, feat_mode_num=self.na_in, num_heads=self.num_heads)
        # elif config.t_method_type == 2: # 0.8475,(xyz + R_i0 * Scalar*delta_T)_mean, current best
        self.outblockRT = M.SO3OutBlockRT(params['outblock'], norm=1, pooling_method='max',
                                              global_scalar=True, use_anchors=True, feat_mode_num=self.na_in, num_heads=self.num_heads)
        # elif config.t_method_type == 3: # (xyz + R_i0 * delta_T)_mean
        #     self.outblockRT = M.SO3OutBlockRT(params['outblock'], norm=1, pooling_method=config.model.pooling_method,
        #                                       feat_mode_num=self.na_in, num_heads=self.num_heads)

        self.anchors = torch.from_numpy(L.get_anchors(60)).cuda()

    def forward(self, x):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        if x.shape[-1] > 3:
            x = x.permute(0, 2, 1).contiguous()
        x = M.preprocess_input(x, self.na_in, False)

        for block_i, block in enumerate(self.backbone):
            x = block(x)
        if self.t_method_type < 0:
            output = self.outblockR(x, self.anchors)
        else:
            output = self.outblockRT(x, self.anchors) # 1, delta_R, deltaT
        output['xyz']     = x.xyz

        return output

    def get_anchor(self):
        return self.backbone[-1].get_anchor()

def build_model(mlps=[[32,32], [64,64], [128,128], [256, 256]],
                out_mlps=[128, 128],
                strides=[2, 2, 2, 1],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.8,
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                sigma_ratio= 0.5, # 1e-3, 0.68
                xyz_pooling = None, # None, 'no-stride'
                to_file=None):

    device = torch.device('cuda:{}'.format(0))
    input_num= 1024 #opt.model.input_num
    dropout_rate= 0.0 #opt.model.dropout_rate
    temperature= 3 #opt.train_loss.temperature
    so3_pooling = 'attention' #opt.model.flag
    input_radius = 0.4 #opt.model.search_radius
    kpconv = False #opt.model.kpconv

    na = 1 if kpconv else 60

    # to accomodate different input_num
    if input_num > 1024:
        sampling_ratio /= (input_num / 1024)
        strides[0] = int(2 * (input_num / 1024))
        print("Using sampling_ratio:", sampling_ratio)
        print("Using strides:", strides)

    print("[MODEL] USING RADIUS AT %f"%input_radius)
    params = {'name': 'Invariant SPConv Model',
              'backbone': [],
              'na': na
              }
    dim_in = 1

    # process args
    n_layer = len(mlps)
    stride_current = 1
    stride_multipliers = [stride_current]
    for i in range(n_layer):
        stride_current *= 2
        stride_multipliers += [stride_current]

    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]

    radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]

    # radius_ratio = [0.25, 0.5]
    radii = [r * input_radius for r in radius_ratio]

    weighted_sigma = [sigma_ratio * radii[0]**2]
    for idx, s in enumerate(strides):
        weighted_sigma.append(weighted_sigma[idx] * s)

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0

            stride_conv = i == 0 or xyz_pooling != 'stride'

            # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))

            if i == 0 and j == 0:
                neighbor *= int(input_num / 1024)

            kernel_size = 1
            if j == 0:
                # stride at first (if applicable), enforced at first layer
                inter_stride = strides[i]
                nidx = i if i == 0 else i+1
                # nidx = i if (i == 0 or xyz_pooling != 'stride') else i+1
                if stride_conv:
                    neighbor *= 2 # * int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
                    kernel_size = 1 # if inter_stride < 4 else 3
            else:
                inter_stride = 1
                nidx = i+1

            # one-inter one-intra policy
            block_type = 'inter_block' if na != 60  else 'separable_block'

            inter_param = {
                'type': block_type,
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size,
                    'stride': inter_stride,
                    'radius': radii[nidx],
                    'sigma': weighted_sigma[nidx],
                    'n_neighbor': neighbor,
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier,
                    'activation': 'leaky_relu',
                    'pooling': xyz_pooling,
                    'kanchor': na,
                }
            }
            block_param.append(inter_param)

            dim_in = dim_out

        params['backbone'].append(block_param)

    representation = 'quat' #opt.model.representation
    params['outblock'] = {
            'dim_in': dim_in,
            'mlp': out_mlps,
            'fc': [64],
            'k': 40,
            'kanchor': na,
            'pooling': so3_pooling,
            'representation': representation,
            'temperature': temperature,
    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = MocapNet(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)

# @hydra.main(config_path="/equi-pose/configs", config_name="pose")
def main():
    BS = 2
    N  = 56 * 64
    C  = 3
    device = torch.device("cuda:0")
    x = torch.randn(BS, N, 3).to(device)
    print("Performing a regression task...")
    model = build_model()
    out = model(x)
    # print(out.feats.shape)
    print(out['R'].shape)
    print(out['T'].shape)
    print(out['1'].shape)
    print(out['xyz'].shape)

if __name__ == '__main__':
    main()
