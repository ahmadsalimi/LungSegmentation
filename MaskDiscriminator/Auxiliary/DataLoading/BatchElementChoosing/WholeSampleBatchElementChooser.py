import numpy as np
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser

'''
This element chooser can be used when we need to choose whole samples.
This element chooser clip by the size of elements per batch parameter for current samples
and if a sample has not enough elements it append -1 at the end of this sample and declare the 
end of this sample in order to replace it with new one.
The difference between this element chooser and CompleteElementsIteratorBatchElementChooser is 
this element chooser choose elements for the permanent batch samples, however the other element chooser, 
replace the finished samples with new ones.
'''


class WholeSampleBatchElementChooser(BatchElementChooser):

    def __init__(self, data_loader):
        super(WholeSampleBatchElementChooser, self).__init__(data_loader)

        self.remain_indices = np.array([])

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

    def reset(self):
        super(WholeSampleBatchElementChooser, self).reset()
        self.remain_indices = np.asarray([])
    
    def finished_iteration_on_previous_batch(self):
        return self.remain_indices == -1

    def prepare_new_batch_element_indices(self, batch_sample_indices):

        elements_per_batch = self.data_loader.conf['elements_per_batch']
        pointer = np.full(batch_sample_indices.shape[0], 0)
        if len(self.remain_indices) == 0:
            self.remain_indices = pointer

        def my_func(xi):
            x = self.prev_batch_sample_indices[xi]
            match_inds = np.where(batch_sample_indices == x)

            if len(match_inds) != 0:
                if self.remain_indices[xi] != -1:
                    pointer[match_inds[0]] = self.remain_indices[xi]

        v_func = np.vectorize(my_func)
        if len(self.prev_batch_sample_indices) > 0:
            v_func(np.arange(len(self.prev_batch_sample_indices)))
        self.remain_indices = pointer

        def choose_element_for_sample(index):

            this_sample_elements_num = len(self.content_loader_of_interest.get_elements_names()[batch_sample_indices[index]])
            indices = self.remain_indices[index]

            if indices != -1:
                start_index, end_index = indices, np.minimum(indices + elements_per_batch, this_sample_elements_num)
                elements_num = end_index - start_index
                if end_index == this_sample_elements_num:
                    self.remain_indices[index] = -1
                else:
                    self.remain_indices[index] = end_index

                return np.concatenate((range(start_index, end_index),
                                       np.full((elements_per_batch - elements_num,), -1)))
            else:
                print('Warning: rechoosing a finished sample for batch')
                return np.full((elements_per_batch,), -1)

        batch_inds = np.stack(tuple([choose_element_for_sample(i)
                                  for i in range(len(batch_sample_indices))]), axis=0)

        # Clipping extra -1s
        sum_of_each_col = np.sum(batch_inds, axis=0)

        # the columns with sum = -1 * n_rows are full -1
        full_empty_cols = np.arange(len(sum_of_each_col))[sum_of_each_col == -1 * batch_inds.shape[0]]

        if len(full_empty_cols) > 0:
            batch_inds = batch_inds[:, :full_empty_cols[0]]

        return batch_inds
