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

def build_model(mlps=[[32,32], [64,64], [128,128], [256, 256]],
                out_mlps=[128, 128],
                strides=[2, 2, 2, 2],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.8,
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                sigma_ratio= 0.5, # 1e-3, 0.68
                xyz_pooling = None, # None, 'no-stride'
    ):

    # number of points
    input_num= 1024 #opt.model.input_num
    dropout_rate= 0.0 #opt.model.dropout_rate
    temperature= 3 #opt.train_loss.temperature
    so3_pooling = 'attention' #opt.model.flag
    input_radius = 0.4 #opt.model.search_radius
    kpconv = False #opt.model.kpconv

    na = 1 if kpconv else 60

    # to accomodate different input_num to 1024
    if input_num > 1024:
        sampling_ratio /= (input_num / 1024)
        strides[0] = int(2 * (input_num / 1024)) # Makes it 1024 / 2 in the first layer
        print("Using sampling_ratio:", sampling_ratio)
        print("Using strides:", strides)

    print("[MODEL] USING RADIUS AT %f"%input_radius)
    params = {'name': 'Invariant SPConv Model',
              'backbone': [],
              'na': na
              }
    dim_in = 1 # Initially only scalar feature

    # process args
    n_layer = len(mlps)
    stride_current = 1
    stride_multipliers = [stride_current]
    # stride_multipliers = [1, 2, 4, 8, 16]
    for i in range(n_layer):
        stride_current *= 2
        stride_multipliers += [stride_current]

    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers] # [1024, 512, 256, 128, 64]

    radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers] # [0.2*1^0.5, 0.2*2^0.5, 0.2*4^0.5, 0.2*8^0.5, 0.2*16^0.5]

    # radius_ratio = [0.25, 0.5]
    radii = [r * input_radius for r in radius_ratio] # radius_ratio * input_radius: increases the search radius as number of points decreases

    weighted_sigma = [sigma_ratio * radii[0]**2]
    # weighted_sigma = s*r2 * stride[i]
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

    return params


def build_model_graph(mlps=[[32,32], [64,64], [128,128], [256, 256]],
                        out_mlps=[128, 128],
                        initial_radius_ratio = 0.2,
                        sampling_ratio = 0.8,
                        sampling_density = 0.5,
                        kernel_density = 1,
                        kernel_multiplier = 2,
                        sigma_ratio= 0.5, # 1e-3, 0.68
    ):

    # number of points
    input_num= 1024 #opt.model.input_num
    dropout_rate= 0.0 #opt.model.dropout_rate
    temperature= 3 #opt.train_loss.temperature
    so3_pooling = 'attention' #opt.model.flag
    input_radius = 0.4 #opt.model.search_radius
    kpconv = False #opt.model.kpconv

    na = 1 if kpconv else 60

    print("[MODEL] USING RADIUS AT %f"%input_radius)
    params = {'name': 'Invariant SPConv Model',
              'backbone': [],
              'na': na
              }
    dim_in = 1 # Initially only scalar feature

    # process args
    n_layer = len(mlps)
    stride_current = 1
    stride_multipliers = [stride_current]
    # stride_multipliers = [1, 2, 4, 8, 16]
    for i in range(n_layer):
        stride_current *= 2
        stride_multipliers += [stride_current]

    # radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers] # [0.2*1^0.5, 0.2*2^0.5, 0.2*4^0.5, 0.2*8^0.5, 0.2*16^0.5]
    
    # Hard coded for now
    radius_ratio = [1.0 for multiplier in stride_multipliers] # [0.2*1^0.5, 0.2*2^0.5, 0.2*4^0.5, 0.2*8^0.5, 0.2*16^0.5]

    # radius_ratio = [0.25, 0.5]
    radii = [r * input_radius for r in radius_ratio] # radius_ratio * input_radius: increases the search radius as number of points decreases

    weighted_sigma = [sigma_ratio * radii[0]**2]
    # weighted_sigma = s*r2 * stride[i]
    for idx in range(len(mlps)):
        weighted_sigma.append(weighted_sigma[idx])

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
            neighbor = 56

            kernel_size = 1
            if j == 0:
                # stride at first (if applicable), enforced at first layer
                nidx = i if i == 0 else i+1
                # nidx = i if (i == 0 or xyz_pooling != 'stride') else i+1
            else:
                nidx = i+1

            # one-inter one-intra policy
            block_type = 'separable_graph_block'

            inter_param = {
                'type': block_type,
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size,
                    'radius': radii[nidx],
                    'sigma': weighted_sigma[nidx],
                    'activation': 'leaky_relu',
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

    return params


class MocapNetOne2All(nn.Module):
    def __init__(self, 
                mlps=[[32,32], [64,64], [128,128], [256, 256]],
                out_mlps=[128, 128],
                strides=[1, 1, 1, 1],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.8,
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                sigma_ratio= 0.5, # 1e-3, 0.68
                xyz_pooling = None, # None, 'no-stride'
    ):
        super(MocapNetOne2All, self).__init__()

        params = build_model(
            mlps=mlps,
            out_mlps=out_mlps,
            strides=strides,
            initial_radius_ratio = initial_radius_ratio,
            sampling_ratio = sampling_ratio,
            sampling_density = sampling_density,
            kernel_density = kernel_density,
            kernel_multiplier = kernel_multiplier,
            sigma_ratio= sigma_ratio, # 1e-3, 0.68
            xyz_pooling = xyz_pooling, # None, 'no-stride'
        )

        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))

        self.na_in = params['na']
        self.invariance = True
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
        return x
        if self.t_method_type < 0:
            output = self.outblockR(x, self.anchors)
        else:
            output = self.outblockRT(x, self.anchors) # 1, delta_R, deltaT
        output['xyz']     = x.xyz

        return output

    def get_anchor(self):
        return self.backbone[-1].get_anchor()


class MocapNetFramePooled(nn.Module):
    def __init__(self, 
                    mlps=[[32,32], [32,32], [32,32], [32, 32]],
                    out_mlps=[128, 128],
                    initial_radius_ratio = 0.2,
                    sampling_ratio = 0.8,
                    sampling_density = 0.5,
                    kernel_density = 1,
                    kernel_multiplier = 2,
                    sigma_ratio= 0.5, # 1e-3, 0.68
    ):
        super(MocapNetFramePooled, self).__init__()

        params = build_model_graph(mlps,
                                    out_mlps,
                                    initial_radius_ratio,
                                    sampling_ratio,
                                    sampling_density,
                                    kernel_density,
                                    kernel_multiplier,
                                    sigma_ratio, # 1e-3, 0.68
        )

        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))

        self.na_in = params['na']
        self.invariance = True
        self.t_method_type = 2 #config.t_method_type

        self.num_heads = 24*64
        self.classifier = None
        self.outblockRT = M.SO3FrameOutBlockRT(params['outblock'], norm=1, pooling_method='mean',
                                              global_scalar=True, use_anchors=True, feat_mode_num=self.na_in, num_heads=self.num_heads)

        self.anchors = torch.from_numpy(L.get_anchors(60)).cuda()

    def forward(self, x):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        x = x.view(-1, x.shape[-2], 3)
        if x.shape[-1] > 3:
            x = x.permute(0, 2, 1).contiguous()
        x = M.preprocess_input(x, self.na_in, False)

        inter_idx, inter_w = None, None
        for block_i, block in enumerate(self.backbone):
            inter_idx, inter_w, x = block(x, inter_idx, inter_w)

        output = self.outblockRT(x, self.anchors) # 1, delta_R, deltaT
        output['xyz']     = x.xyz

        return output

    def get_anchor(self):
        return self.backbone[-1].get_anchor()

# @hydra.main(config_path="/equi-pose/configs", config_name="pose")
def main():
    BS = 1
    T = 64
    N  = 56
    C  = 3
    device = torch.device("cuda")
    x = torch.randn(BS, T, N, 3).to(device)
    print("Performing a regression task...")
    model = MocapNetFramePooled().to(device)
    out = model(x)
    # print(out.feats.shape)
    # print(out['R'].shape)
    # print(out['T'].shape)
    # print(out['1'].shape)
    # print(out['xyz'].shape)

if __name__ == '__main__':
    main()
