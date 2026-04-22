# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:13:00 2026

@author: efrat.sasson
"""

import h5py
import torch
from torch.utils.data import Dataset

class PCamH5Dataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.x_file = h5py.File(x_path, "r")
        self.y_file = h5py.File(y_path, "r")

        self.x = self.x_file["x"]   # shape: (N, 96, 96, 3)
        self.y = self.y_file["y"]   # shape: (N, 1) usually
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]          # uint8, HWC
        label = int(self.y[idx][0])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        return img, label
    
    