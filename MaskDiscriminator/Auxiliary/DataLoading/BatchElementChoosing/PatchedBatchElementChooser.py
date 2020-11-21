from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser
from Auxiliary.DataLoading.DataLoader import DataLoader
from abc import abstractmethod
import numpy as np
from Auxiliary.DataLoading.ContentLoading.BoolLobeMaskLoader import BoolLobeMaskLoader
from typing import Dict

class TrainPatchedBatchElementChooser(BatchElementChooser):
    
    def __init__(self, data_loader: DataLoader):
        """ Receives data loader which is a class containing information about the samples
        and sets it in itself for using. """
        super().__init__(data_loader)
        self.patch_count = self.data_loader.conf['patch_count']
        self.patch_height = self.data_loader.conf['patch_height']
        self.bool_mask_content_loader: BoolLobeMaskLoader = \
            self.data_loader.get_content_loader_of_interest(BoolLobeMaskLoader)

    def finished_iteration_on_previous_batch(self):
        """ Returns a boolean array with the same size of batch,
        for each sample of the batch would be True if the batch element chooser
        has done iterating over the sample and false if not. """
        return np.full(len(self.prev_batch_sample_indices), True)
    
    def extract_patches(self, height: int) -> np.ndarray:
        max_choice = 1 + max(0, height - self.patch_height)
        patch_starts: np.ndarray = np.random.choice(max_choice, size=self.patch_count, replace=max_choice < self.patch_count)
        heights = np.full(self.patch_count, self.patch_height)
        return np.stack((patch_starts, heights), axis=1)

    def prepare_new_batch_element_indices(self, batch_sample_indices: np.ndarray) -> np.ndarray:
        """ Receives the sample indices for the new batch, sets up whatever needed in itself
        and returns the indices of the chosen elements for each sample for the next batch. """
        heights = self.bool_mask_content_loader.get_samples_heights()
        return np.stack(tuple(self.extract_patches(heights[i]) for i in batch_sample_indices), axis=0)
        

class PatchData:

    def __init__(self, height: int, patch_count: int, patch_height: int):
        starts = np.arange(start=0, stop=1 + max(0, height - patch_height), step=patch_height // 2)

        while starts.shape[0] % patch_count != 0:
            pad = patch_count - starts.shape[0] % patch_count
            starts = np.concatenate((starts, starts[:pad]))
        
        heights = np.full(starts.shape[0], patch_height)
        self.patch = np.stack((starts, heights), axis=1)
        
        self.patch_count = patch_count
        self.current_index = 0
    
    def finished(self) -> bool:
        return self.current_index == self.patch.shape[0]
    
    def next(self) -> np.ndarray:
        if self.finished():
            raise Exception("No more elements available")

        self.current_index += self.patch_count
        return self.patch[self.current_index-self.patch_count:self.current_index]


class TestPatchedBatchElementChooser(BatchElementChooser):

    def __init__(self, data_loader: DataLoader):
        """ Receives data loader which is a class containing information about the samples
        and sets it in itself for using. """
        super().__init__(data_loader)
        self.patch_count = self.data_loader.conf['patch_count']
        self.patch_height = self.data_loader.conf['patch_height']
        self.bool_mask_content_loader: BoolLobeMaskLoader = \
            self.data_loader.get_content_loader_of_interest(BoolLobeMaskLoader)
        self.patches: Dict[int, PatchData] = {}


    def finished_iteration_on_previous_batch(self):
        """ Returns a boolean array with the same size of batch,
        for each sample of the batch would be True if the batch element chooser
        has done iterating over the sample and false if not. """
        return np.array([s_index not in self.patches for s_index in self.prev_batch_sample_indices])
    
    def extract_patches(self, index: int, height: int) -> np.ndarray:
        if index not in self.patches:
            self.patches[index] = PatchData(height, self.patch_count, self.patch_height)

        patch_data = self.patches[index]

        patch = patch_data.next()
        if patch_data.finished():
            self.patches.pop(index)
        return patch

    def prepare_new_batch_element_indices(self, batch_sample_indices: np.ndarray) -> np.ndarray:
        """ Receives the sample indices for the new batch, sets up whatever needed in itself
        and returns the indices of the chosen elements for each sample for the next batch. """
        heights = self.bool_mask_content_loader.get_samples_heights()
        return np.stack(tuple(self.extract_patches(i, heights[i]) for i in batch_sample_indices), axis=0)