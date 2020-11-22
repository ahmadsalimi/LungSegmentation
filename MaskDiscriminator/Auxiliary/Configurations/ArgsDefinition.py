import argparse


def define_arg_parser():
    parser = argparse.ArgumentParser(description='Predicting Corona.')

    parser.add_argument('-device', dest='device',
                        default='cpu', type=str,
                        help='Analysis will run over device, use cpu, cuda:#, cuda (for all gpus)',
                        metavar='device')

    parser.add_argument('-phase', dest='phase',
                        default='train', type=str,
                        help='Phase of the analysis, can be train, eval, evalisolated, eval2, save',
                        metavar='phase')

    parser.add_argument('-samples_dir', dest='samples_dir',
                        default=None, type=str,
                        help='The directory/dataGroup to be evaluated. In the case of dataGroup can be train/test/val. In the case of directory must contain 0 and 1 subdirectories. Use:FilterName to do over samples containing /FilterName/ in their path',
                        metavar='samples_dir')

    parser.add_argument('-load_dir', dest='load_dir',
                        default=None, type=str,
                        help='The directory to load the model from, when not given will be calculated!',
                        metavar='load_dir')

    parser.add_argument('-save_dir', dest='save_dir',
                        default=None, type=str,
                        help='The directory to save the model from, when not given will be calculated!',
                        metavar='save_dir')

    parser.add_argument('-report_dir', dest='report_dir',
                        default=None, type=str,
                        help='The dir to save reports per slice per sample in.',
                        metavar='report_dir')

    parser.add_argument('-epoch', dest='epoch',
                        default=None, type=str,
                        help='The epoch to load.',
                        metavar='epoch')

    parser.add_argument('-try_num', dest='try_num',
                        default=None, type=int,
                        help='The try number to load',
                        metavar='try_num')

    parser.add_argument('-dataSeparation', dest='dataSeparation',
                        default=None, type=str,
                        help='The data_separation to be used.',
                        metavar='dataSeparation')

    parser.add_argument('-batch_size', dest='batch_size',
                        default=None, type=int,
                        help='The batch size to be used.',
                        metavar='batch_size')

    parser.add_argument('-elements_per_batch', dest='elements_per_batch',
                        default=None, type=int,
                        help='The number of elements in batch.',
                        metavar='elements_per_batch')

    parser.add_argument('-evaluator', dest='evaluator',
                        default=None, type=str,
                        help='The evaluator to be used.',
                        metavar='evaluator')

    parser.add_argument('-LabelMapDict', dest='LabelMapDict',
                        default=None, type=(lambda x: get_dict_from_str(x)),
                        help='The dictionary used for mapping the labels to the ones used for the model.',
                        metavar='LabelMapDict')

    parser.add_argument('-debug_mode', dest='debug_mode',
                        default=None, type=(lambda x: x == '1'),
                        help='Whether in debug mode!',
                        metavar='debug_mode')

    parser.add_argument('-augment', dest='augment',
                        default=None, type=(lambda x: x == '1'),
                        help='Whether use augmentation in training/evaluation!',
                        metavar='augment')

    parser.add_argument('-pretrained_model_file', dest='pretrained_model_file',
                        default=None, type=str,
                        help='Address of .pt pretrained model',
                        metavar='pretrained_model_file')

    return parser


def get_dict_from_str(dstr):
    the_dict = dict()

    for kvp in dstr.split(','):
        k, v = kvp.split(':')
        the_dict[int(k)] = int(v)

    return the_dict

