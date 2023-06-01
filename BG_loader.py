from glob import glob
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import re
import numpy as np
import os
import random as rd
from scipy.io import loadmat


class BG_Dataset_Labeled(Dataset):
    def __init__(self, root_path, mat_file, type_data, src_dim):
        raw_data = loadmat("%s/%s" % (root_path, mat_file))
        self.id_data = torch.from_numpy(raw_data["%s_id" % type_data])
        try:
            self.grad_data = torch.from_numpy(raw_data["%s_gradient" % type_data])
        except:
            print("[warning] no gradient data for data type %s" % type_data)
            self.grad_data = torch.from_numpy(raw_data["%s_para" % type_data])
        self.para_data = torch.from_numpy(raw_data["%s_para" % type_data])
        self.force_data = torch.from_numpy(raw_data["%s_force" % type_data])
        self.disp_data = torch.from_numpy(raw_data["%s_disp" % type_data])
        self.series_data = torch.from_numpy(raw_data["%s_series" % type_data].T)

        # print(self.id_data.shape, self.grad_data.shape, self.para_data.shape, self.force_data.shape, self.disp_data.shape, self.series_data.shape)

        self.len = self.force_data.size(0)
        assert self.len == self.para_data.size(0)

        self.force_data = self.force_data.reshape(self.force_data.size(0), -1, src_dim)
        self.disp_data = self.disp_data.reshape(self.disp_data.size(0), -1, src_dim)
        self.id_data = self.id_data.int()
        self.series_data = self.series_data.int()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.force_data[index], self.para_data[index], self.grad_data[index], self.disp_data[index], self.id_data[index]