from Auxiliary.AllClassesStats import *
from Auxiliary.Configurations.Configs import get_save_dir, update_config_based_on_args
from os import path, makedirs


def run_main(conf, model):

    update_config_based_on_args(conf)

    # Define the required directories if they're not given
    if conf['save_dir'] is None:
        conf['save_dir'] = get_save_dir(conf)
    if conf['load_dir'] is None:
        conf['load_dir'] = get_save_dir(conf)

    # checking if the required directories does not exist, creating them
    if not path.exists(conf['save_dir']):
        makedirs(conf['save_dir'])

    # Printing information about the run:
    print('@@@')
    print('Phase: ', conf['phase'])
    print('Load dir: ', conf['load_dir'])
    print('Save dir: ', conf['save_dir'])
    print('@@@')

    # Checking if the mode is in debug mode, seizing inputs!
    if conf['debug_mode']:
        conf['batch_size'] = 3
        if conf['elements_per_batch'] >= 3:
            conf['elements_per_batch'] = 3
        conf['val_iters_per_epoch'] = 2
        conf['dataSeparation'] = 'SmallTest'
        conf['big_batch_size'] = 2

    if type(conf['evaluator']) == str:
        conf['evaluator'] = get_evals_dict()[conf['evaluator']]

    phase_obj = get_phases_dict()[conf['phase']](conf, model)
    phase_obj.run()
