import glob
import pandas as pd
import fnmatch
import re
import random
import os
import sklearn
import numpy as np


def get_by_label(root_dir, label):
    files = list(set(glob.glob(os.path.join(root_dir, "**", str(label), "**/*.npy")) + glob.glob(os.path.join(root_dir, "**", str(label), "*.npy"))))
    labels = [bool(label)] * len(files)

    np.random.shuffle(files)

    train_count = int(np.ceil(len(files) * .7))
    test_count = int(np.ceil(len(files) * .2))
    valid_count = len(files) - train_count - test_count

    groups = ["Train"] * train_count + ["Test"] * \
        test_count + ["Valid"] * valid_count

    assert len(files) == len(labels) and len(files) == len(groups)

    return files, labels, groups


def split_datasets(root_dir):

    abnormal_files, abnormal_Labels, abnormal_groups = get_by_label(root_dir, 2)
    positive_files, positive_labels, positive_groups = get_by_label(root_dir, 1)
    negative_files, negative_labels, negative_groups = get_by_label(root_dir, 0)

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
    root_dir = '/home/ghavami.ce.sharif/MaskDiscriminator/Data'
    csv_file = '/home/ghavami.ce.sharif/MaskDiscriminator/data_split_01.csv'
    df = split_datasets(root_dir)
    df.to_csv(csv_file, index=False)
