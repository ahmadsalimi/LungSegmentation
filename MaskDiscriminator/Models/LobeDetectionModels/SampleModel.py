import torch
import torch.backends.cudnn
from torch.nn import Module
import numpy as np # linear algebra
from os import getcwd
from sys import path as sys_path
sys_path.insert(0, getcwd())
from Auxiliary.Main import run_main
from Auxiliary.Configurations.Configs import get_shared_config
from MiddleProcessing.NeuronInfectionMasking.ThresholdedNeuronInfectionMasker import ThresholdedInfectionNeuronMasker
from Auxiliary.ModelEvaluation.CovidEvaluators.BinaryCovidEvaluator import BinaryCovidEvaluator
from Auxiliary.ModelTraining.Trainer import Trainer
from Auxiliary.ModelRunning.ModelRunner import ModelRunner
from Auxiliary.DataLoading.BatchChoosing.RandomBatchChooser import RandomBatchChooser
from Auxiliary.DataLoading.BatchChoosing.SequentialBatchChooser import SequentialBatchChooser
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from Auxiliary.DataLoading.BatchElementChoosing.RandomOffsetInHeightBatchElementChooser import RandomOffsetInHeightBatchElementChooser
from Auxiliary.DataLoading.BatchElementChoosing.CompleteElementsIteratorBatchElementChooser import CompleteElementsIteratorBatchElementChooser
from Models.CoronaModels.IT_HamedNet2_6 import CoronaPredictor as IT_HamedNet2_6
from Auxiliary.DataLoading.BatchChoosing.BalancedBatchChooser import BalancedBatchChooser
from Auxiliary.DataLoading.BatchElementChoosing.PreciseEquallySpacedSliceChooser import PreciseEquallySpacedSliceChooser


def get_configs():

    n_conf = {
        'debug_mode': False,
        'init_lr': 1e-6,
        'big_batch_size': 128,
        'batch_size': 4,
        'elements_per_batch': 7,
        'try_num': 601,
        'try_name': 'ThickNet',
        'max_epochs': 100,
        'turn_on_val_mode': True,
        '3_slice_samples': True,
        'patch_labler': ThresholdedInfectionNeuronMasker('M1', 28, 36, 14, 32, 1.0/3),
        'subsample': False,
        'dataSeparation': 'v10DMV_AbnTRCless',
        'distMaskDirGiver': (lambda lobe_dir: lobe_dir.replace('../', '../NeuronDistMask/M1/')),
        'iters_per_epoch': 10,
        'val_iters_per_epoch': 100,
        'trainer': Trainer,
        'evaluator': BinaryCovidEvaluator,
        'model_runner': ModelRunner,
        'content_loaders': [(CovidCTLoader, {'prefix_name': 'ct'})],
        'train_sampler': (BalancedBatchChooser, PreciseEquallySpacedSliceChooser),
        'val_sampler': (BalancedBatchChooser, PreciseEquallySpacedSliceChooser),
        'test_sampler': (SequentialBatchChooser, CompleteElementsIteratorBatchElementChooser),
        'LabelMapDict': {0: 0, 1: 1, 2: 1}
    }

    m_conf = get_shared_config()
    m_conf.update(n_conf)

    return m_conf


if __name__ == '__main__':

    the_conf = get_configs()
    model = CoronaPredictor()
    run_main(the_conf, model)
