from Auxiliary.RunsPhases.Phase import Phase
from Auxiliary.ModelLoading.ModelLoading import load_best_model
from time import time
import torch


class SavePhase(Phase):

    def __init__(self, conf, model):
        super(SavePhase, self).__init__('save', conf, model)

    def run_subclass(self):
        """ Loads the model from the specified epoch and directory and
        saves it as an independent file."""

        # save the model for the specified epochs
        self.run_for_multiple_epochs(self.save_model)

    def save_model(self):
        """ Saves the model for a single epoch. """

        if self.conf['load_dir'] != self.conf['save_dir']:
            save_name = '%s_e%s.plt' % (self.conf['load_dir'], self.conf['epoch'])
        else:
            save_name = '%s_%d_e%s.pt' % (self.conf['try_name'], self.conf['try_num'], self.conf['epoch'])

        # loading the model to save
        load_best_model(self.model, self.conf)
        torch.save(self.model.state_dict(), save_name)
        print('Model saved successfully in %s' % save_name)
