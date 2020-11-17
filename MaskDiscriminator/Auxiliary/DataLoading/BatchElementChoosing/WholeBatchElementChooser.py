from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser
from Auxiliary.DataLoading.DataLoader import DataLoader
from abc import abstractmethod
import numpy as np
from Auxiliary.DataLoading.ContentLoading.BoolLobeMaskLoader import BoolLobeMaskLoader


class TrainWholeBatchElementChooser(BatchElementChooser):

    def __init__(self, data_loader: DataLoader):
        """ Receives data loader which is a class containing information about the samples
        and sets it in itself for using. """
        super().__init__(data_loader)
        self.bool_mask_content_loader: BoolLobeMaskLoader = \
            self.data_loader.get_content_loader_of_interest(BoolLobeMaskLoader)

    def finished_iteration_on_previous_batch(self):
        """ Returns a boolean array with the same size of batch,
        for each sample of the batch would be True if the batch element chooser
        has done iterating over the sample and false if not. """
        return np.full(len(self.prev_batch_sample_indices), True)

    def extract_patches(self, height: int) -> np.ndarray:
        return np.array([[0, height]])

    def prepare_new_batch_element_indices(self, batch_sample_indices: np.ndarray) -> np.ndarray:
        """ Receives the sample indices for the new batch, sets up whatever needed in itself
        and returns the indices of the chosen elements for each sample for the next batch. """
        heights = self.bool_mask_content_loader.get_samples_heights()
        return np.stack(tuple(self.extract_patches(heights[i]) for i in batch_sample_indices), axis=0)
        
