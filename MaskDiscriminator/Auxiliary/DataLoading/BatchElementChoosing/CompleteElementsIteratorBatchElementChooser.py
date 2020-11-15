from abc import abstractmethod
import numpy as np
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser


'''
This element chooser can be used when we need to choose all the elements of samples.
This element chooser clip by the size of elements per batch parameter for current samples
and if a sample has not enough elements it append -1 at the end of this sample and declare the 
end of this sample in order to replace it with new one.
This element chooser is used for the test mode, mostly.
'''


class CompleteElementsIteratorBatchElementChooser(BatchElementChooser):

    def __init__(self, data_loader):
        super(CompleteElementsIteratorBatchElementChooser, self).__init__(data_loader)

        self.return_pointer = np.array([])

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

    def reset(self):
        super(CompleteElementsIteratorBatchElementChooser, self).reset()
        self.return_pointer = np.asarray([])

    @abstractmethod
    def finished_iteration_on_previous_batch(self):
        return self.return_pointer == -1

    def prepare_new_batch_element_indices(self, batch_sample_indices):
        elements_per_batch = self.data_loader.conf['elements_per_batch']
        pointer = np.full(batch_sample_indices.shape[0], 0)
        if len(self.return_pointer) == 0:
            self.return_pointer = pointer

        def my_func(xi):
            x = self.prev_batch_sample_indices[xi]
            match_inds = np.where(batch_sample_indices == x)

            if len(match_inds) != 0:
                if self.return_pointer[xi] != -1:
                    pointer[match_inds[0]] = self.return_pointer[xi]

        v_func = np.vectorize(my_func)
        if len(self.prev_batch_sample_indices) > 0:
            v_func(np.arange(len(self.prev_batch_sample_indices)))
        self.return_pointer = pointer

        def choose_element_for_sample(index):
            sample_index = batch_sample_indices[index]
            num_elements = len(self.content_loader_of_interest.get_elements_names()[sample_index])
            start_index = self.return_pointer[index]

            if start_index + elements_per_batch == num_elements:
                self.return_pointer[index] = -1
                return np.arange(start_index, start_index + elements_per_batch)

            elif start_index + elements_per_batch < num_elements:
                self.return_pointer[index] = start_index + elements_per_batch
                return np.arange(start_index, start_index + elements_per_batch)

            else:
                self.return_pointer[index] = -1
                return np.concatenate((np.arange(start_index, num_elements),
                                      np.full((elements_per_batch - (num_elements - start_index),), -1)), axis=0)

        ret_val = [choose_element_for_sample(i) for i in range(len(batch_sample_indices))]
        ret_val = np.stack(tuple(ret_val), axis=0)
        return ret_val
