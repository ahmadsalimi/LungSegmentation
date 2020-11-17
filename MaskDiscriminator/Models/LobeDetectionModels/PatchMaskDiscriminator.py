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
from Auxiliary.DataLoading.BatchElementChoosing.PatchedBatchElementChooser import TrainPatchedBatchElementChooser
from Models.Model import Model
from typing import Union, Tuple, Iterable, Dict
from Models.LobeDetectionModels.commons import ResidualBlock
import torch.nn.functional as F


class PatchMaskDiscriminator(Model):

    def __init__(self, batch_norms: Tuple[bool, bool, bool, bool, bool, bool, bool, bool]):
        super().__init__()

        self.conv = nn.Sequential(                                                                                                  # B 3   64  256 256
            ResidualBlock((3, 4, 8), 2, 2, 4, 4, padding=0, batch_norms=batch_norms[:2]),                                           # B 8   16  64  64
            ResidualBlock((8, 16, 32), 2, 2, 4, 4, padding=0, batch_norms=batch_norms[2:4]),                                        # B 32  4   16  16
            ResidualBlock((32, 64, 128), 2, 2, 4, 4, padding=0, batch_norms=batch_norms[4:6]),                                      # B 128 1   4   4
            ResidualBlock((128, 256, 512), (1, 2, 2), (1, 2, 2), (1, 4, 4), (1, 4, 4), padding=0, batch_norms=batch_norms[6:]),     # B 512 1   1   1
        )

        self.decider = nn.Sequential(                       # B 512
            nn.Linear(512, 256),                            # B 256
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(256, 128),                            # B 128
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(128, 1),                              # B 1
            nn.Sigmoid()
        )

    def forward(self, x_sample: torch.Tensor, x_label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # x     B   P   3   64  256 256

        B, P = x_sample.shape[:2]

        x = x_sample.flatten(0, 1)  # B*P   3   64  256 256

        out = self.conv(x)          # B*P   512 1   1   1
        out = out.reshape(-1, 512)  # B*P   512
        out = self.decider(out)     # B*P   1
        out = out.reshape(B, P)     # B     P
        out = out.amax(dim=1)       # B

        if x_label is None:
            return {'positive_class_probability': out}

        loss = F.binary_cross_entropy(out, x_label)

        return {
            'positive_class_probability': out,
            'loss': loss
        }





def get_configs():

    n_conf = {
        'debug_mode': True,
        'init_lr': 1e-4,
        'big_batch_size': 1,
        'batch_size': 4,
        'patch_count': 5,
        'patch_height': 64,
        'try_num': 1,
        'try_name': 'PatchBatchDiscriminator',
        'max_epochs': 100,
        'dataSeparationDir': './',
        'dataSeparation': 'data_split_01',
        'iters_per_epoch': 64,
        'val_iters_per_epoch': 16,
        'trainer': Trainer,
        'evaluator': BinaryEvaluator,
        'model_runner': ModelRunner,
        'content_loaders': [(BoolLobeMaskLoader, {'prefix_name': 'x'})],
        'train_sampler': (RandomBatchChooser, TrainPatchedBatchElementChooser),
        'val_sampler': (RandomBatchChooser, TrainPatchedBatchElementChooser),
        'test_sampler': (SequentialBatchChooser, TrainPatchedBatchElementChooser),
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
    model = PatchMaskDiscriminator(the_conf['batch_norms'])
    run_main(the_conf, model)
