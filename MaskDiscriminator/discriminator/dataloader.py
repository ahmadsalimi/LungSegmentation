import torch
import numpy as np
from os import path
import glob
import torch.nn.functional as F
import pandas as pd
from collections.abc import Iterable
import sklearn


def load_data(samples:pd.DataFrame, batch_size:int) -> tuple[list[np.ndarray], np.ndarray]:

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

def get_data_loader(data_split_file: str, group: str, batch_size:int, sample_count: int=-1) -> Iterable[tuple[list[np.ndarray], np.ndarray]]:
    data_split: pd.DataFrame = pd.read_csv(data_split_file)
    positive_samples:pd.DataFrame = data_split[data_split.Label]
    negative_samples:pd.DataFrame = data_split[~data_split.Label]

    if sample_count > 0:
        positive_samples = positive_samples.sample(n=sample_count // 2)
        negative_samples = negative_samples.sample(n=sample_count // 2)
    
    samples = pd.concat((positive_samples, negative_samples))
    samples = sklearn.utils.shuffle(samples)

    return enumerate(load_data(samples, batch_size))