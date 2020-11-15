from abc import abstractmethod
from Auxiliary.DataLoading.ContentLoading.BoolLobeMaskLoader import BoolLobeMaskLoader
import numpy as np


class BatchElementChooser:

    def __init__(self, data_loader: BoolLobeMaskLoader):
        """ Receives data loader which is a class containing information about the samples
        and sets it in itself for using. """
        self.data_loader: BoolLobeMaskLoader = data_loader
        self.prev_batch_sample_indices = np.asarray([])

    @abstractmethod
    def finished_iteration_on_previous_batch(self):
        """ Returns a boolean array with the same size of batch,
        for each sample of the batch would be True if the batch element chooser
        has done iterating over the sample and false if not. """

    def get_new_batch_element_indices(self, batch_sample_indices):
        """ Receives the sample indices for the new batch, sets up whatever needed in itself
        and returns the indices of the chosen elements for each sample for the next batch. """
        new_elements_inds = self.prepare_new_batch_element_indices(batch_sample_indices)
        self.prev_batch_sample_indices = batch_sample_indices
        return new_elements_inds

    @abstractmethod
    def prepare_new_batch_element_indices(self, batch_sample_indices):
        """ Receives the sample indices for the new batch, sets up whatever needed in itself
        and returns the indices of the chosen elements for each sample for the next batch. """

    def reset(self):
        """ Resets the element iterator. """
        self.prev_batch_sample_indices = np.asarray([])
