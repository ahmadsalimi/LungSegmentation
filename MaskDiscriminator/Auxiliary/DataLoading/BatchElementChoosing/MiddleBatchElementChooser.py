from abc import abstractmethod
import numpy as np
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser

'''
This element chooser can be used when we need to choose the part of samples.
This element chooser select the middle section of elements of each sample based on the element_per_batch 
parameter.
'''


class MiddleSampleBatchElementChooser(BatchElementChooser):

    def __init__(self, data_loader):
        super(MiddleSampleBatchElementChooser, self).__init__(data_loader)

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

    @abstractmethod
    def finished_iteration_on_previous_batch(self):
        return np.full((len(self.prev_batch_sample_indices),), True)

    def prepare_new_batch_element_indices(self, batch_sample_indices):

        elements_per_batch = self.data_loader.conf['elements_per_batch']

        def choose_element_for_sample(index):

            this_sample_elements_num = len(
                self.content_loader_of_interest.get_elements_names()[batch_sample_indices[index]])

            middle = int(np.floor(this_sample_elements_num / 2))
            start_index = np.maximum(0, middle - int(np.floor(1.0 * elements_per_batch / 2)))
            end_index = np.minimum(middle + int(np.ceil(1.0 * elements_per_batch / 2)), this_sample_elements_num)
            elements_num = end_index - start_index

            if elements_num == elements_per_batch:
                return range(start_index, end_index)
            else:
                return np.concatenate((range(start_index, end_index),
                                       np.full((elements_per_batch - elements_num,), -1)))

        batch_inds = np.stack(tuple([choose_element_for_sample(i)
                                     for i in range(len(batch_sample_indices))]), axis=0)

        # Clipping extra -1s
        sum_of_each_col = np.sum(batch_inds, axis=0)

        # the columns with sum = -1 * n_rows are full -1
        full_empty_cols = sum_of_each_col == -1 * batch_inds.shape[0]

        batch_inds = batch_inds[:, np.logical_not(full_empty_cols)]
        return batch_inds
