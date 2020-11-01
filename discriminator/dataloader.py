import torch
import numpy as np
from tqdm import tqdm
from os import path
import glob
import torch.nn.functional as F


def load_data(files, shuffle=False):
    if shuffle:
        np.random.shuffle(files)

    for file in files:
        sample = torch.tensor(np.load(file))
        
        assert sample.ndim == 4
        assert sample.shape[0] == 3
        assert sample.shape[-1] == 512
        assert sample.shape[-2] == 512

        sample = F.interpolate(sample.float(), 256, mode='bilinear', align_corners=False).numpy()[np.newaxis]
        label = np.array([bool(int(file.split("/")[-2]))])

        yield sample, label

def get_data_loader(root_path, data_for, shuffle=False):
    files = glob.glob(path.join(root_path, data_for, "**/*.npy"), recursive=True)
    num_batches = len(files)
    return tqdm(
        enumerate(load_data(files, shuffle=shuffle)),
        position=0,
        leave=True,
        desc=data_for,
        total=num_batches)