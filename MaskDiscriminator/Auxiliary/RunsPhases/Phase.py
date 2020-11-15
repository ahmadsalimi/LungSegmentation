from abc import abstractmethod
from time import time
import traceback
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader


class Phase:

    def __init__(self, name, conf, model):
        """ Receives as input the name of the phase and the configurations"""

        self.name = name
        self.conf = conf
        self.model = model

    def extract_epochs(self):
        """ Returns the epochs specified for the phase to be run in """

        if 'epoch' not in self.conf:
            return None

        epochs_to_eval = []
        for ep in self.conf['epoch'].split(','):
            if ':' not in ep:
                epochs_to_eval.append(ep)
            else:
                epochs_to_eval += range(int(ep.split(':')[0]), int(ep.split(':')[1]))

        return epochs_to_eval

    def extract_sample_groups(self):
        """ Returns the list of sample groups for running the phase for them"""

        return self.conf['samples_dir'].split(',')

    def run_for_multiple_epochs(self, func):
        """ Runs the given function in the specified epochs.
        The function must be wrapped in a lambda expression to have no inputs
        so it can be called in a generic mode."""

        # if epoch is not defined (in which case the model should be imported from a pt file), running in one epoch
        if self.conf['epoch'] is None:
            func()
        else:

            # Keeping a backup of epoch as we are going to overwrite it
            ep_bu = self.conf['epoch']

            for ep in self.extract_epochs():
                try:
                    print('\n>>>>>>> epoch %s' % ep)
                    self.conf['epoch'] = ep
                    func()
                except Exception as e:
                    print('Problem in epoch ', ep)
                    track = traceback.format_exc()
                    print(track)

            self.conf['epoch'] = ep_bu

    def run(self):
        """ Runs the suitable function for the phase implemented in subclasses,
        times it and prints the total time spent on the phase. """
        t1 = time()
        self.run_subclass()
        t2 = time()
        print('Phase %s done in %.2f secs.' % (self.name, t2 - t1))

    @abstractmethod
    def run_subclass(self):
        """ This function would be run when the user specifies the phase"""

    def instantiate_trainer(self):
        """ Instantiates a trainer object to run training with, and returns it """
        return self.conf['trainer'](self.model, self.conf)

    def instantiate_evaluator(self, data_loader):
        """ Instantiates an evaluator object to run evaluation with, and returns it """
        return self.conf['evaluator'](self.model, self.conf, data_loader)

    def close_thread_pools(self, data_loader):
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                cl.loader_workers.finish()