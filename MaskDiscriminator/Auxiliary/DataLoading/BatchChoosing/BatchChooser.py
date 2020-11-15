from abc import abstractmethod
import numpy as np


class BatchChooser:

    def __init__(self, config, data_loader, batch_element_chooser=None):
        """ Receives as input config, which is a dictionary containing configurations of the run
        and batch_element_chooser which is an instance of the type BatchElementChooser,
        if set would be responsible of choosing sample elements for each batch.
        For samples that doesn't have elements can be set to None to have no effect. """

        self.dataLoader = data_loader
        self.config = config
        self.batch_element_chooser = batch_element_chooser

        self.current_batch_sample_indices = np.asarray([], dtype=int)
        self.current_batch_elements_indices = None

        self.completed_iteration = False

        self.class_samples_indices = self.dataLoader.class_samples_indices
        self.batch_size = config['batch_size']

        # in the case of receiving a list of (prob, element chooser) to choose between multiple options!
        self.element_choosers_probs = None
        if self.batch_element_chooser is not None:
            if type(self.batch_element_chooser) == list:
                self.element_choosers_probs = np.asarray([x[0] for x in self.batch_element_chooser])
                self.batch_element_chooser = [x[1] for x in self.batch_element_chooser]
            else:
                self.element_choosers_probs = np.asarray([1])
                self.batch_element_chooser = [self.batch_element_chooser]

    def prepare_next_batch(self):
        """ Chooses samples and slices of those samples for the next batch of the run,
        saves their indices to prepare the required information about the current batch anytime needed. """

        element_chooser = None
        if self.element_choosers_probs is not None:
            element_chooser = self.batch_element_chooser[
                np.argmax(np.random.multinomial(1, self.element_choosers_probs, 1))]

        # Checking if done iterating over the samples
        sample_completion_flags = np.full((len(self.current_batch_sample_indices),), True)
        if element_chooser is not None:
            sample_completion_flags = \
                element_chooser.finished_iteration_on_previous_batch()

        #print('@@@@')
        #print(sample_completion_flags)

        self.current_batch_sample_indices = \
            self.get_next_batch_sample_indices(sample_completion_flags).astype(int)

        #print(self.current_batch_sample_indices)

        if len(self.current_batch_sample_indices) == 0:
            self.completed_iteration = True
            return

        # choosing elements if element chooser is not None
        if element_chooser is not None:
            self.current_batch_elements_indices = \
                element_chooser.get_new_batch_element_indices(self.current_batch_sample_indices).astype(int)

        #print(self.current_batch_elements_indices)

    @abstractmethod
    def get_next_batch_sample_indices(self, sample_completion_flags):
        """ Receives sample_completion_flags which is a numpy array of boolean
        True if done iterating the sample, false otherwise. If False the sample must be kept
        in the next batch. Returns a list containing the indices of the samples chosen for the next batch."""

    def get_current_batch_sample_indices(self):
        """ Returns a list of indices of the samples chosen for the current batch. """
        return self.current_batch_sample_indices

    def get_current_batch_elements_indices(self):
        """ Returns a list of lists, one list per sample containing lists of the chosen elements
         of the samples chosen for the current batch. """
        return self.current_batch_elements_indices

    def finished_iteration(self):
        """ Returns True if iteration is finished over all the slices of all the samples, False otherwise"""
        return self.completed_iteration

    def reset(self):
        """ Resets the sample iterator. """
        self.completed_iteration = False
        self.current_batch_sample_indices = np.asarray([], dtype=int)
        self.current_batch_elements_indices = None
        if self.batch_element_chooser is not None:
            for x in self.batch_element_chooser:
                x.reset()
