from abc import abstractmethod
import numpy as np
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser


'''
The same as RandomOffsetInHeightBatchElementChooser!
Only this one iterates for all the possible offsets for each batch.
Returns False when iteration is done on all possible offsets.
'''


class SequentialOffsetInHeightBatchElementChooser(BatchElementChooser):

    def __init__(self, data_loader):
        super(SequentialOffsetInHeightBatchElementChooser, self).__init__(data_loader)

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

        self.next_offset = np.asarray([])
        self.current_samples_partition_sizes = np.asarray([])

    @abstractmethod
    def finished_iteration_on_previous_batch(self):
        if len(self.next_offset) == 0:
            return np.asarray([])
        return self.next_offset == self.current_samples_partition_sizes

    def prepare_new_batch_element_indices(self, batch_sample_indices):

        elements_per_batch = self.data_loader.conf['elements_per_batch']
        next_offset = np.full(batch_sample_indices.shape[0], 0)
        current_samples_partition_sizes = np.full(batch_sample_indices.shape[0], -1)

        if len(self.next_offset) == 0:
            self.next_offset = next_offset
            self.current_samples_partition_sizes = current_samples_partition_sizes

        def my_func(xi):
            x = self.prev_batch_sample_indices[xi]
            match_inds = np.where(batch_sample_indices == x)

            if len(match_inds) != 0:
                if self.next_offset[xi] != self.current_samples_partition_sizes[xi]:
                    next_offset[match_inds[0]] = self.next_offset[xi]
                    current_samples_partition_sizes[match_inds[0]] = self.current_samples_partition_sizes[xi]

        v_func = np.vectorize(my_func)
        if len(self.prev_batch_sample_indices) > 0:
            v_func(np.arange(len(self.prev_batch_sample_indices)))

        self.next_offset = next_offset
        self.current_samples_partition_sizes = current_samples_partition_sizes

        def choose_elements_for_sample(index):
            num_elements = len(self.content_loader_of_interest.get_elements_names()[batch_sample_indices[index]])

            if num_elements <= elements_per_batch:
                self.current_samples_partition_sizes[index] = 1
                r_val = np.concatenate((range(0, num_elements),
                                       np.full((elements_per_batch - num_elements,), num_elements - 1)), axis=0)
            else:
                if self.current_samples_partition_sizes[index] == -1:
                    self.current_samples_partition_sizes[index] = \
                        int(np.ceil(1.0 * num_elements / elements_per_batch))
                random_offset = self.next_offset[index]
                els = random_offset + self.current_samples_partition_sizes[index] * np.arange(elements_per_batch)
                els[els >= num_elements] = num_elements - 1
                r_val = els

            self.next_offset[index] += 1
            return r_val

        ret_val = np.stack(tuple([choose_elements_for_sample(i) for i in range(len(batch_sample_indices))]), axis=0)

        return ret_val

    def reset(self):
        super(SequentialOffsetInHeightBatchElementChooser, self).reset()
        self.next_offset = np.asarray([])
        self.current_samples_partition_sizes = np.asarray([])