from abc import abstractmethod
import numpy as np
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser


'''
This element chooser segments the elements based on elements per batch
parameter and return elements with particular distances(elements per batch).
This element chooser use for the train mode when the elements of samples have large lengths, mostly.
'''


class RandomOffsetInHeightBatchElementChooser(BatchElementChooser):

    def __init__(self, data_loader):
        super(RandomOffsetInHeightBatchElementChooser, self).__init__(data_loader)

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

    @abstractmethod
    def finished_iteration_on_previous_batch(self):
        return np.full(len(self.prev_batch_sample_indices), True)

    def prepare_new_batch_element_indices(self, batch_sample_indices):

        elements_per_batch = self.data_loader.conf['elements_per_batch']

        def choose_elements_for_sample(index):
            num_elements = len(self.content_loader_of_interest.get_elements_names()[batch_sample_indices[index]])

            if num_elements <= elements_per_batch:
                return np.concatenate((range(0, num_elements),
                                       np.full((elements_per_batch - num_elements,), num_elements - 1)), axis=0)
            else:
                partition_size = int(np.ceil(1.0 * num_elements / elements_per_batch))
                random_offset = np.random.randint(0, partition_size - 1, 1)
                els = random_offset + partition_size * np.arange(elements_per_batch)
                els[els >= num_elements] = num_elements - 1
                return els

        ret_val = np.stack(tuple([choose_elements_for_sample(i) for i in range(len(batch_sample_indices))]), axis=0)

        return ret_val


''' OLD
            start_index = 0
            if num_elements == elements_per_batch:
                return range(start_index, num_elements)
            elif num_elements < elements_per_batch:
                return np.concatenate((range(start_index, num_elements),
                                       np.full((elements_per_batch - num_elements,),
                                               num_elements - 1)), axis=0)
            else:
                start_index = np.random.randint(np.ceil(num_elements / elements_per_batch))

                def create(x):
                    return x * (np.ceil(num_elements / elements_per_batch)) + start_index

                l_func = np.vectorize(create)
                element_slices = l_func(np.arange(elements_per_batch))
                element_slices[-1] = np.minimum(num_elements - 1, element_slices[-1])
                return element_slices
            '''
