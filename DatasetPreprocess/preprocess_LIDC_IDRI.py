import pylidc as pl
import numpy as np
import os
import matplotlib.pyplot as plt
from LungDetection2 import MaskPreprocessor
from scipy import ndimage
from multiprocessing import Pool, Manager
from contextlib import redirect_stdout
from sys import argv
import time


def create_output_dirs(config):
    for dirname in [config['scans_dirname'], config['masks_dirname']]:
        path = os.path.join(config['output_dir'], dirname)
        os.makedirs(path, exist_ok=True)


def get_nudules_masks(scan, shape):
    nodules_mask = np.zeros(shape, dtype='bool')

    for annotation in scan.annotations:
        nodule_mask = annotation.boolean_mask().transpose(2, 0, 1)
        bbox = annotation.bbox()
        bbox = tuple(bbox[i] for i in (2, 0, 1))

        nodules_mask[bbox] = nodule_mask

    return nodules_mask

def save_sample(config, patient_id, scan, masks, spacing):
    scan_path = os.path.join(config['output_dir'], config['scans_dirname'], f"LIDC-IDRI_{patient_id.split('-')[-1]}_{spacing}")
    np.save(scan_path, scan)
    mask_path = os.path.join(config['output_dir'], config['masks_dirname'], f"LIDC-IDRI_{patient_id.split('-')[-1]}")
    np.save(mask_path, masks)

def get_scans(config):
    return map(lambda scan: scan.patient_id, pl.query(pl.Scan).filter(pl.Scan.patient_id <= f"LIDC-IDRI-{config['last_id']:04d}"))

def preprocess_sample(config, patient_id):
    start_time = time.time()

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    print(f"{patient_id}: Start preprocessing", flush=True)

    vol = scan.to_volume().transpose(2, 0, 1).clip(-1024, 3071)[::-1]
    spacing = f"{scan.pixel_spacing}_{scan.pixel_spacing}_{scan.slice_thickness}"

    print(f"{patient_id}: {{shape: {vol.shape}, spacing: {spacing}}}", flush=True)

    nodules_mask = get_nudules_masks(scan, vol.shape)
    segmentor = MaskPreprocessor()
    masks = segmentor.preprocess(vol, scan, nodules_mask)

    print(f"{patient_id}: lung mask created at {time.time() - start_time:.2f}s!", flush=True)

    save_sample(config, patient_id, vol, masks, spacing)

def logger(file, q):
    while True:
        m = q.get()
        if m == 'end':
            print("ended", file=file, flush=True)
            break
        
        print(m, file=file, flush=True)

if __name__ == '__main__':
    if len(argv) != 2:
        print("Usage: python preprocess_LIDC_IDRI.py config_file")
    else:
        config_file = argv[1]
        with open(config_file) as configuration:
            exec(configuration.read())
        
        pool = Pool(processes)
        
        create_output_dirs(config)
        
        with redirect_stdout(log_file):
            jobs = []
            for patient_id in get_scans(config):
                job = pool.apply_async(preprocess_sample, (config, patient_id))
                jobs.append(job)

            for job in jobs:
                job.get()

        pool.close()
        pool.join()