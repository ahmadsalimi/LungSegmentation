from abc import abstractmethod
import numpy as np
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader, get_next_slice_index, get_prev_slice_index
from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser


'''
This element chooser segments the elements based on elements per batch
parameter and return elements with particular distances(elements per batch).
This element chooser use for the train mode when the elements of samples have large lengths, mostly.
'''


class InterestedSliceBasedSelector(BatchElementChooser):

    def __init__(self, data_loader, wrong_slice_choosing_prob=0):
        super(InterestedSliceBasedSelector, self).__init__(data_loader)

        self.wrong_slice_choosing_prob = wrong_slice_choosing_prob

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

        # filtering the slices not of interest!
        samples_labels = self.content_loader_of_interest.samples_labels
        slices_probs = self.content_loader_of_interest.load_slice_probs(
            self.data_loader.conf['slice_probs_dir'])
        self.slices_of_interest = []

        for i in range(len(samples_labels)):
            # for healthy all are interested!
            if samples_labels[i] == 0:
                self.slices_of_interest.append(np.arange(len(slices_probs[i])))
            else:
                # otherwise only the diseased slices
                d_slices = np.arange(len(slices_probs[i]))[slices_probs[i] >= 0.5]
                # the top diseased slice and the two neighbors of that must also be inside!
                top_slice = np.argmax(slices_probs[i])

                top_slices = np.unique(np.asarray([
                    get_prev_slice_index(self.content_loader_of_interest.samples_slices_paths, top_slice),
                    top_slice,
                    get_next_slice_index(self.content_loader_of_interest.samples_slices_paths, top_slice),
                ]))

                d_slices = np.unique(np.concatenate((d_slices, top_slices), axis=0))
                self.slices_of_interest.append(d_slices)

    @abstractmethod
    def finished_iteration_on_previous_batch(self):
        return np.full(len(self.prev_batch_sample_indices), True)

    def prepare_new_batch_element_indices(self, batch_sample_indices):

        elements_per_batch = self.data_loader.conf['elements_per_batch']

        def choose_elements_for_sample(index):

            sample_index = batch_sample_indices[index]

            if self.wrong_slice_choosing_prob > 0 and \
                np.random.binomial(1, self.wrong_slice_choosing_prob, 1) == 1:

                sample_els_probs = (((
                    self.data_loader.model_preds_for_smaples_elements[sample_index])[
                    self.slices_of_interest[sample_index]]) >= 0.5).astype(int)

                wrong_els_inds = (self.slices_of_interest[sample_index])[
                    sample_els_probs != self.content_loader_of_interest.samples_labels[sample_index]]

                if len(wrong_els_inds) > 0:
                    els = wrong_els_inds[
                        np.random.randint(0, len(wrong_els_inds), size=(elements_per_batch,))
                    ]

                    return els

            els = (self.slices_of_interest[sample_index])[
                np.random.randint(0, len(self.slices_of_interest[sample_index]), size=(elements_per_batch,))]

            return els

        ret_val = np.stack(tuple([choose_elements_for_sample(i) for i in range(len(batch_sample_indices))]), axis=0)

        return ret_val

