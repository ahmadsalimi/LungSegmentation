from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser
from Auxiliary.DataLoading.ContentLoading.BoolLobeMaskLoader import BoolLobeMaskLoader
from abc import abstractmethod
import numpy as np


class TrainPatchedBatchElementChooser(BatchElementChooser):

    def __init__(self, data_loader: BoolLobeMaskLoader):
        """ Receives data loader which is a class containing information about the samples
        and sets it in itself for using. """
        super().__init__(data_loader)
        self.patch_count = self.data_loader.conf['patch_count']
        self.patch_height = self.data_loader.conf['patch_height']

    def finished_iteration_on_previous_batch(self):
        """ Returns a boolean array with the same size of batch,
        for each sample of the batch would be True if the batch element chooser
        has done iterating over the sample and false if not. """
        return np.full(len(self.prev_batch_sample_indices), True)

    def extract_patches(self, height: int) -> np.ndarray:
        max_choice = max(1, height - self.patch_height)
        patch_starts: np.ndarray = np.random.choice(max_choice, size=self.patch_count, replace=max_choice < self.patch_count)
        heights = np.full(self.patch_count, self.patch_height)
        return np.stack((patch_starts, height), axis=1)

    def prepare_new_batch_element_indices(self, batch_sample_indices: np.ndarray) -> np.ndarray:
        """ Receives the sample indices for the new batch, sets up whatever needed in itself
        and returns the indices of the chosen elements for each sample for the next batch. """
        heights = self.data_loader.get_samples_heights()
        return np.stack(tuple(self.extract_patches(heights[i]) for i in batch_sample_indices), axis=0)
        
