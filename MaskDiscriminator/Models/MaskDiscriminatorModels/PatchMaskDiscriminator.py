import torch.backends.cudnn
from torch import nn
from os import getcwd
from sys import path as sys_path
sys_path.insert(0, getcwd())
from Auxiliary.Main import run_main
from Auxiliary.Configurations.Configs import get_shared_config
from Auxiliary.ModelEvaluation.BinaryEvaluator import BinaryEvaluator
from Auxiliary.ModelTraining.Trainer import Trainer
from Auxiliary.ModelRunning.ModelRunner import ModelRunner
from Auxiliary.DataLoading.BatchChoosing.SequentialBatchChooser import SequentialBatchChooser
from Auxiliary.DataLoading.ContentLoading.BoolLobeMaskLoader import BoolLobeMaskLoader
from Auxiliary.DataLoading.BatchChoosing.RandomBatchChooser import RandomBatchChooser
from Auxiliary.DataLoading.BatchElementChoosing.PatchedBatchElementChooser import TrainPatchedBatchElementChooser, TestPatchedBatchElementChooser
from Models.MaskDiscriminatorModels.models import PatchMaskDiscriminator


def get_configs():

    n_conf = {
        'debug_mode': False,
        'init_lr': 1e-4,
        'big_batch_size': 4,
        'batch_size': 2,
        'patch_count': 3,
        'patch_height': 64,
        'try_num': 5,
        'try_name': 'PatchBatchDiscriminator',
        'max_epochs': 100,
        'dataSeparationDir': '../',
        'dataSeparation': 'data_split_01',
        'iters_per_epoch': 32,
        'val_iters_per_epoch': 32,
        'trainer': Trainer,
        'evaluator': BinaryEvaluator,
        'model_runner': ModelRunner,
        'content_loaders': [(BoolLobeMaskLoader, {'prefix_name': 'x'})],
        'train_sampler': (RandomBatchChooser, TrainPatchedBatchElementChooser),
        'val_sampler': (RandomBatchChooser, TrainPatchedBatchElementChooser),
        'test_sampler': (SequentialBatchChooser, TestPatchedBatchElementChooser),
        'LabelMapDict': {True: 1, False: 0},
        'batch_norms': (
            False, True, True, True, False, False, False, False
        ),
        'console_file': 'eval_console.log'
    }

    m_conf = get_shared_config()
    m_conf.update(n_conf)

    return m_conf


if __name__ == '__main__':

    the_conf = get_configs()
    model = PatchMaskDiscriminator(the_conf['batch_norms'])
    run_main(the_conf, model)
