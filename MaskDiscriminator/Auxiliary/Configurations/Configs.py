from Auxiliary.Configurations.ArgsDefinition import define_arg_parser
import random
import torch
from sys import stderr
import numpy as np
import os
#import imgaug as aug
from os import path, makedirs


# SHARED CONFIGS
def get_shared_config():
    m_conf = {
        'inp_size': 256,
        'elements_per_batch': 10,
        'batch_size': 8,
        'big_batch_size': 1,
        'm_seed': 17,
        'max_epochs': 10,
        'patches_cnt_in_row': 32,
        '3_slice_samples': False,
        'patch_labler': None,
        'subsample': False,
        'dataSeparation': None,
        'distMaskDirGiver': None,
        'iters_per_epoch': None,
        'val_iters_per_epoch': None,
        'train_sampler_as_val_sampler': False,
        'needs_sample_filtering': False,
        'needs_couple_model': False,
        'debug_mode': False,
        'augmenter': None,
        'require_grad_for_input': False,
        'phase': 'train',
        'samples_dir': None,
        'load_dir': None,
        'save_dir': None,
        'report_dir': None,
        'epoch': None,
        'repeat_value': 1,
        'augment': False,
        'track_preds_on_data': False,
        'turn_on_val_mode': True,
    }

    args = define_arg_parser().parse_args()
    m_conf['args'] = args
    m_conf['dev_name'], m_conf['device'] = get_device(args)

    return m_conf


def set_random_seeds(the_seed):

    # Set the seed for hash based operations in python
    os.environ['PYTHONHASHSEED'] = '0'

    torch.manual_seed(the_seed)
    np.random.seed(the_seed)
    random.seed(the_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the numpy seed
    np.random.seed(the_seed)

    # Make the augmentation sequence deterministic
    #aug.seed(the_seed)


# returns device and device name
def get_device(args):

    # cuda num, -1 means parallel, -2 means cpu, no cuda means cpu
    dev_name = args.device

    if 'cuda' in dev_name:
        gpu_num = 0
        if ':' in dev_name:
            gpu_num = int(dev_name.split(':')[1])

        if gpu_num >= torch.cuda.device_count():
            print('No %s, using CPU instead!' % dev_name, file=stderr)
            dev_name = 'cpu'
            the_device = torch.device(dev_name)
            print('@ Running on CPU @')

    if dev_name == 'cuda':
        dev_name = None
        the_device = torch.device('cuda')
        print('@ Running on all %d GPUs @' % torch.cuda.device_count())

    elif 'cuda:' in dev_name:
        the_device = torch.device(dev_name)
        print('@ Running on GPU:%d @' % int(dev_name.split(':')[1]))

    else:
        dev_name = 'cpu'
        the_device = torch.device(dev_name)
        print('@ Running on CPU @')

    return dev_name, the_device


def get_save_dir(conf):

    save_dir = '../Results/%d_%s_%d' % (
        conf['try_num'], conf['try_name'], conf['m_seed'])

    save_dir += '_%s' % conf['dataSeparation']

    if not path.exists(save_dir):
        makedirs(save_dir)
    return save_dir


def update_config_based_on_args(the_conf):

    args = the_conf['args']
    args_attrs = dir(args)

    for k in the_conf.keys():
        if k == 'args':
            continue
        if k in args_attrs and getattr(args, k) is not None:
            the_conf[k] = getattr(args, k)

    print('@@@@@@@@@@@@@@@@@@')
    print('The used config is: ' , the_conf)
    #for k, v in the_conf.items():
    #    print(k, ': ', v)
    print('@@@@@@@@@@@@@@@@@@')

    '''
        m_conf['phase'] = args.phase
    
        m_conf['samples_dir'] = args.samples_dir
    
        m_conf['load_dir'] = args.load_dir
        m_conf['save_dir'] = args.save_dir
        m_conf['report_dir'] = args.report_dir
        m_conf['epoch'] = args.epoch
        m_conf['couple_model_dir'] = args.couple_model_dir
        m_conf['tryn'] = args.tryn
        m_conf['city'] = args.city
        m_conf['n_batch_size'] = args.batch_size
        m_conf['n_data_separation'] = args.data_separation
        m_conf['n_evaluator'] = args.evaluator
    '''

    # setting the device
