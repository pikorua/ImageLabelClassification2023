import torch
from torch.utils.data import Dataset
import fnmatch
import os
import numpy as np




class customDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, index):
        label = torch.load(self.label_dir + '/' + str(index + 1) + '.pt')
        image = torch.load(self.image_dir + '/' + str(index + 1) + '.pt')
        sample = {'image': image, 'label': label}
        return sample




