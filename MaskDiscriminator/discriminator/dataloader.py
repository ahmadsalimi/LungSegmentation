import torch
import numpy as np
from os import path
import glob
import torch.nn.functional as F


def load_data(files):

    for file in files:
        file = file.replace("\\","/")
        sample = torch.tensor(np.load(file))

        assert sample.ndim == 4
        assert sample.shape[0] == 3
        assert sample.shape[-1] == 512
        assert sample.shape[-2] == 512

        sample = F.interpolate(sample.float(), 256, mode='bilinear', align_corners=False).numpy()[np.newaxis]
        label = np.array([bool(int(file.split("/")[-2]))])

        yield sample, label

def get_data_loader(root_path, data_for, sample_per_epoch = -1, shuffle=False):
    positive_files = glob.glob(path.join(root_path, data_for, "1/*.npy"), recursive=True)
    negative_files = glob.glob(path.join(root_path, data_for, "0/*.npy"), recursive=True)
    if shuffle:
        positive_files = np.random.choice(positive_files, size=sample_per_epoch//2, replace=False)
        negative_files = np.random.choice(negative_files, size=sample_per_epoch//2, replace=False)
    
    files = np.concatenate((positive_files, negative_files))
    np.random.shuffle(files)
    return enumerate(load_data(files))