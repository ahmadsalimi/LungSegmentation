import pandas as pd
import numpy as np
from contextlib import redirect_stdout
from typing import Iterable

files: Iterable[str] = pd.read_csv("clip_files.csv")['0'].values

for file in files:
    sample: np.ndarray = np.load(file)
    print(f"clipping file {file.split('/')[-1]} with min: {sample.min()} and max: {sample.max()}")
    sample = sample.clip(-1024, 4095-1024)
    save_path = file[:-4]
    np.save(save_path, sample)
    print(f"file {save_path.split('/')[-1]}.npy saved!")
    print("-----------------------------------------------------")
