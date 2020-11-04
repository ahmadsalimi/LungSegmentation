import os
import glob
from os import path
import random

for label in [0, 1]:
    files = glob.glob(path.join("MaskDiscriminatorData", "**", str(label), "**/*.npy"), recursive=True)
    train_count = len(files) * .7
    valid_count = len(files) * .1
    
    random.shuffle(files)
    for i, file in enumerate(files):
        if i < train_count:
            os.rename(file, f"Data/Train/{label}/{i:03d}.npy")
        elif i < train_count + valid_count:
            os.rename(file, f"Data/Valid/{label}/{i:03d}.npy")
        else:
            os.rename(file, f"Data/Test/{label}/{i:03d}.npy")