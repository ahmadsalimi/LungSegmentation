import pylidc as pl
import numpy as np
import os
import matplotlib.pyplot as plt
from LungDetection2 import MaskPreprocessor
from scipy import ndimage

def draw(images, columns=4):
    rows = int(np.ceil(images.shape[0] / columns))
    max_size = 20
    
    width = max(columns * 5, max_size)
    height = width * rows // columns

    plt.figure(figsize=(width, height))
    plt.gray()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(images.shape[0]):
        plt.subplot(rows,columns,i+1), plt.imshow(images[i]), plt.axis('off')
        # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
    plt.show()


last_id = 1
output_dir = "./"

for folder in ["scans", "masks"]:
    path = os.path.join(output_dir, folder)
    if not os.path.isdir(path):
        os.mkdir(path)

for scan in pl.query(pl.Scan).filter(pl.Scan.patient_id <= f'LIDC-IDRI-{last_id:04d}'):
    print(f"Start preprocessing {scan.patient_id}")

    vol = scan.to_volume().transpose(2, 0, 1).clip(-1024, 3071)
    spacing = f"{scan.pixel_spacing}_{scan.pixel_spacing}_{scan.slice_thickness}"

    print(f"shape: {vol.shape} - spacing: {spacing}")

    full_mask = np.zeros(vol.shape, dtype='bool')

    for annotation in scan.annotations:
        nodule_mask = annotation.boolean_mask().transpose(2, 0, 1)
        bbox = annotation.bbox()
        bbox = tuple(bbox[i] for i in (2, 0, 1))

        full_mask[bbox] = nodule_mask

    segmentor = MaskPreprocessor()

    masks = segmentor.preprocess(vol, scan, full_mask)

    print(f"lung mask created!")

    draw(masks, columns=10)