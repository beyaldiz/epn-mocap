import numpy as np
import trimesh
import os
import glob
import scipy.io as sio
import torch
import torch.utils.data as data
import __init__
import vgtk.pc as pctk
import vgtk.so3conv.functional as L
from vgtk.functional import rotation_distance_np
import random
import pathlib
import multiprocessing
from tqdm import tqdm

n_anchor = 60
def bp():
    import pdb;pdb.set_trace()


def read_file_ours(file_path):
    data = np.load(file_path)

    # Raw markers
    raw_markers = data["M1"].astype(np.float32)

    # Global joint positions
    J_t = data["J_t"].astype(np.float32)
    J_R = data["J_R"].astype(np.float32)

    anc_label = data["anc_label"].astype(np.int)
    rot_from_anc = data["rot_from_anc"].astype(np.float32)

    return (
        raw_markers,
        J_t,
        J_R,
        anc_label,
        rot_from_anc
    )

class MS_Synthetic(data.Dataset):
    def __init__(self, data_dir='data/', mode=None):
        super(MS_Synthetic, self).__init__()

        # 'train' or 'eval'
        # self.mode = opt.mode if mode is None else mode
        self.mode = mode

        if 'val' in self.mode:
            self.mode = 'test'
        files_dir = (
            pathlib.Path(data_dir)
            / "ours_Synthetic"
            / ("msalign_train_epn_data" if self.mode == 'train' else "msalign_test_epn_data")
        )
        file_paths = [f for f in files_dir.glob("*.npz")]
        print(f"{len(file_paths)} animation files found in {files_dir}")
        self.file_paths = file_paths

        # Store all things in ram
        # Load data in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            data_list = pool.map(read_file_ours, file_paths)

        print("Read done!")

        raw_markers = np.concatenate([d[0] for d in data_list], axis=0)
        print("Raw markers concatenated!")
        J_t = np.concatenate([d[1] for d in data_list], axis=0)
        print("J_t concatenated!")
        J_R = np.concatenate([d[2] for d in data_list], axis=0)
        print("J_R concatenated!")
        anc_label = np.concatenate([d[3] for d in data_list], axis=0)
        print("anc_label concatenated!")
        rot_from_anc = np.concatenate([d[4] for d in data_list], axis=0)
        print("rot_from_anc concatenated!")


        del data_list

        self.raw_markers = raw_markers
        self.J_t = J_t
        self.J_R = J_R
        self.anc_label = anc_label
        self.rot_from_anc = rot_from_anc


    def __len__(self):
        return self.raw_markers.shape[0]

    def __getitem__(self, index):
        return {'xyz': torch.from_numpy(self.raw_markers[index]),
                'J_R': torch.from_numpy(self.J_R[index]),
                'J_t': torch.from_numpy(self.J_t[index]),
                'R_label': torch.from_numpy(self.anc_label[index]),
                'R_rel': torch.from_numpy(self.rot_from_anc[index]),
        }

if __name__ == '__main__':
    opt = None
    dset = MS_Synthetic(mode='train')
    data = dset.__getitem__(0)
    print(data['xyz'].shape)
    print(data['J_R'].shape)
    print(data['J_t'].shape)
    print(data['R_label'].shape)
    print(data['R_rel'].shape)
