import numpy as np
import glob


datasets = ['LUNA16', 'VESSEL12']

for dataset in datasets:
    for file in glob.glob(f"{dataset}/**/*.npy"):
        print(f"reversing file {file.split('/')[-1]}")
        sample = np.load(file)
        sample = sample[::-1]
        save_path = file[:-4]
        np.save(save_path, sample)
        print(f"file {save_path.split('/')[-1]}.npy saved!")
        print("---------------------------------------------")