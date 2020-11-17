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
from Auxiliary.DataLoading.BatchElementChoosing.WholeBatchElementChooser import TrainWholeBatchElementChooser
from Models.MaskDiscriminatorModels.models import WholeMaskDiscriminator


def get_configs():

    n_conf = {
        'debug_mode': False,
        'init_lr': 1e-4,
        'big_batch_size': 8,
        'batch_size': 1,
        'try_num': 1,
        'try_name': 'WholeBatchDiscriminator',
        'max_epochs': 100,
        'dataSeparationDir': './',
        'dataSeparation': 'SmallTest',
        'iters_per_epoch': 1,
        'val_iters_per_epoch': 1,
        'trainer': Trainer,
        'evaluator': BinaryEvaluator,
        'model_runner': ModelRunner,
        'content_loaders': [(BoolLobeMaskLoader, {'prefix_name': 'x'})],
        'train_sampler': (RandomBatchChooser, TrainWholeBatchElementChooser),
        'val_sampler': (RandomBatchChooser, TrainWholeBatchElementChooser),
        'test_sampler': (SequentialBatchChooser, TrainWholeBatchElementChooser),
        'LabelMapDict': {True: 1, False: 0},
        'batch_norms': (
            False, True, True, True, False, False, False, False
        )
    }

    m_conf = get_shared_config()
    m_conf.update(n_conf)

    return m_conf


if __name__ == '__main__':

    the_conf = get_configs()
    model = WholeMaskDiscriminator(the_conf['batch_norms'])
    run_main(the_conf, model)
