import numpy as np
from os import path
import torch


def load_best_model(the_model, conf):

    dev_name = conf['dev_name']

    if (conf['load_dir'] == conf['save_dir']) and ('final_version' in conf):
        load_dir = conf['final_version']
        if not path.exists(load_dir):
            raise Exception('No path %s' % load_dir)
    else:
        load_dir = conf['load_dir']

    print('>>> loading from: ' + load_dir)

    map_location = None
    if dev_name == 'cpu':
        map_location = 'cpu'
    elif dev_name is not None and ':' in dev_name:
        map_location = dev_name

    if dev_name != 'cpu':
        if dev_name is not None:
            the_model.cuda(int(dev_name.split(':')[1]))
        else:
            the_model.cuda()

    if path.isfile(load_dir):

        the_model.load_state_dict(torch.load(load_dir, map_location=map_location))
        print('Loaded the model at %s' % load_dir)

    elif path.exists(load_dir + '/GeneralInfo'):

        epoch = int(conf['epoch'])
        print('Loading model from %s' % load_dir)

        if epoch is None:
            val_evals = np.load(load_dir + '/val_evals.npy')
            val_evals = val_evals[val_evals[:, 0] > 0]

            epoch = np.argmin(val_evals[:, 0])
            print('Loaded the model with lowest validation loss, epoch: %d' % epoch)
        else:
            print('Loaded the model at epoch: %d' % epoch)

        checkpoint = torch.load('%s/%d' % (load_dir, epoch), map_location=map_location)
        the_model.load_state_dict(checkpoint['model_state_dict'])

    else:
        print('%s does not exist' % (load_dir + '/GeneralInfo',))

    if dev_name != 'cpu':

        if dev_name is not None:
            the_model.cuda(int(dev_name.split(':')[1]))
        else:
            the_model = torch.nn.DataParallel(the_model)
            the_model.cuda()

    return the_model


def load_couple_model(couple_model, conf):

    dev_name = conf['dev_name']
    save_dir = conf['couple_model_dir']

    map_location = None
    if dev_name == 'cpu':
        map_location = 'cpu'
    elif dev_name is not None and ':' in dev_name:
        map_location = dev_name

    if dev_name != 'cpu':
        if dev_name is not None:
            couple_model.cuda(int(dev_name.split(':')[1]))
        else:
            couple_model.cuda()

    couple_model.load_state_dict(torch.load(save_dir, map_location=map_location))
    print('Loaded the couple model from %s' % save_dir)

    return couple_model
