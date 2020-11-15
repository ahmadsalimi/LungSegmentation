import glob
import pandas as pd
import fnmatch
import re
import random
import os
import sklearn
import numpy as np


def get_by_label(root_dir, label, train_frac, test_frac):
    files = list(set(glob.glob(os.path.join(root_dir, "**", str(label), "**/*.npy")) + glob.glob(os.path.join(root_dir, "**", str(label), "*.npy"))))
    labels = [bool(label)] * len(files)

    np.random.shuffle(files)

    train_count = int(np.ceil(len(files) * train_frac))
    test_count = int(np.floor(len(files) * test_frac))
    valid_count = len(files) - train_count - test_count

    groups = ["Train"] * train_count + ["Test"] * test_count + ["Valid"] * valid_count

    assert len(files) == len(labels) and len(files) == len(groups)

    return files, labels, groups


def split_datasets(root_dir, train_frac, test_frac):

    abnormal_files, abnormal_Labels, abnormal_groups = get_by_label(root_dir, 2, train_frac, test_frac)
    positive_files, positive_labels, positive_groups = get_by_label(root_dir, 1, train_frac, test_frac)
    negative_files, negative_labels, negative_groups = get_by_label(root_dir, 0, train_frac, test_frac)

    print(f"abnormal count: {len(abnormal_files)}")
    print(f"positive count: {len(positive_files)}")
    print(f"negative count: {len(negative_files)}")

    all_files = abnormal_files + positive_files + negative_files
    all_labels = abnormal_Labels + positive_labels + negative_labels
    all_groups = abnormal_groups + positive_groups + negative_groups

    df = pd.DataFrame(data={
        "Group": all_groups,
        "Path": all_files,
        "Label": all_labels
    })

    df = sklearn.utils.shuffle(df)

    return df


if __name__ == "__main__":
    train_frac = .5
    test_frac = 0

    root_dir = 'MaskDiscriminator/Data'
    csv_file = 'MaskDiscriminator/data_split_01.csv'
    df = split_datasets(root_dir, train_frac, test_frac)
    df.to_csv(csv_file, index=False)
