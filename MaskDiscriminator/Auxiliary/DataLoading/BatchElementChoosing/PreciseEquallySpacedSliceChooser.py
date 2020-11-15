from abc import abstractmethod
import numpy as np
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser


class PreciseEquallySpacedSliceChooser(BatchElementChooser):

    def __init__(self, data_loader):
        super(PreciseEquallySpacedSliceChooser, self).__init__(data_loader)

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
            sample_slice_inds = np.arange(len(
                self.content_loader_of_interest.sample_slices_left_lobe_indices[sample_index]))

            if len(sample_slice_inds) < elements_per_batch:
                if len(sample_slice_inds) > 4:
                    sample_slice_inds = np.concatenate((
                        sample_slice_inds[0:2],
                        np.repeat(
                            sample_slice_inds[2:-2],
                            int(np.ceil(1.0 * (elements_per_batch - 4) / (len(sample_slice_inds) - 4)))),
                        sample_slice_inds[-2:]
                    ), axis=0)
                else:
                    sample_slice_inds = np.repeat(
                            sample_slice_inds,
                            int(np.ceil(1.0 * elements_per_batch / len(sample_slice_inds))))

            partitions_starts = np.round(
                np.arange(elements_per_batch) * 1.0 * len(sample_slice_inds) / elements_per_batch).\
                astype(int)

            partition_ends = np.concatenate((
                partitions_starts[1:], np.asarray([len(sample_slice_inds)])), axis=0)

            partition_sizes = partition_ends - partitions_starts

            max_ps = np.amax(partition_sizes)

            offset = np.random.randint(0, max_ps, 1)
            els = np.minimum(partitions_starts + offset, partition_ends - 1)
            els = sample_slice_inds[els]

            # now concatenating the related left and right indices!
            els = np.concatenate((
                self.content_loader_of_interest.sample_slices_left_lobe_indices[sample_index][els],
                self.content_loader_of_interest.sample_slices_right_lobe_indices[sample_index][els],
            ), axis=0)

            return els

        ret_val = np.stack(tuple([choose_elements_for_sample(i) for i in range(len(batch_sample_indices))]), axis=0)

        return ret_val
