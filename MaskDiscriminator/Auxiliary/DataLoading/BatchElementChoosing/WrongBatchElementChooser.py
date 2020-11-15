from abc import abstractmethod
import numpy as np
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser


'''
This element chooser segments the elements based on elements per batch
parameter and return elements with particular distances(elements per batch).
This element chooser use for the train mode when the elements of samples have large lengths, mostly.
'''


class WrongElementChooser(BatchElementChooser):

    def __init__(self, data_loader):
        super(WrongElementChooser, self).__init__(data_loader)

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

            sample_index = batch_sample_indices[index]

            num_elements = len(self.content_loader_of_interest.get_elements_names()[sample_index])

            if num_elements <= elements_per_batch:
                els = np.concatenate((range(0, num_elements),
                                       np.full((elements_per_batch - num_elements,), num_elements - 1)), axis=0)
            else:
                worst_element_ind = np.argmax(self.content_loader_of_interest.slices_probs[sample_index])

                left_els = min(worst_element_ind, int((elements_per_batch - 1) // 2))
                right_els = min(elements_per_batch - 1 - left_els, num_elements - 1 - worst_element_ind)

                # if there is a mismatch with the number = we've had less elements from right => updating left
                if left_els + right_els != elements_per_batch - 1:
                    left_els = elements_per_batch - 1 - right_els

                els = np.arange(worst_element_ind - left_els, worst_element_ind + right_els + 1)
                if len(els) != elements_per_batch:
                    raise Exception('The code in here is buggy!!!')

            els = els.astype(int)
            #print('\tSample %d: ' % sample_index, els)
            #print('\tSample %d: ' % sample_index, self.content_loader_of_interest.slices_probs[sample_index][els])
            return els

        ret_val = np.stack(tuple([choose_elements_for_sample(i) for i in range(len(batch_sample_indices))]), axis=0)

        return ret_val

