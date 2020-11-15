from abc import abstractmethod
import numpy as np
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from Auxiliary.DataLoading.BatchElementChoosing.BatchElementChooser import BatchElementChooser


class SequentialPreciseEquallySpacedSliceChooser(BatchElementChooser):

    def __init__(self, data_loader):
        super(SequentialPreciseEquallySpacedSliceChooser, self).__init__(data_loader)

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

        self.current_samples_next_offset = np.asarray([])
        self.current_sample_slice_inds = []
        self.current_samples_partitions_starts = []
        self.current_samples_partitions_ends = []
        self.current_samples_max_partition_size = np.asarray([])

    @abstractmethod
    def finished_iteration_on_previous_batch(self):
        if len(self.current_samples_next_offset) == 0:
            return np.asarray([])
        return self.current_samples_next_offset == self.current_samples_max_partition_size

    def prepare_new_batch_element_indices(self, batch_sample_indices):

        elements_per_batch = self.data_loader.conf['elements_per_batch']

        next_samples_next_offset = np.zeros((len(batch_sample_indices),), dtype=int)
        next_sample_slices_inds = [None for _ in range(len(batch_sample_indices))]
        next_samples_partitions_starts = [None for _ in range(len(batch_sample_indices))]
        next_samples_partitions_ends = [None for _ in range(len(batch_sample_indices))]
        next_samples_max_partition_size = np.zeros((len(batch_sample_indices),), dtype=int)

        def replace_remaining_samples_info(xi):

            # if sample is finished but rechosen for the batch! It should be reset.
            if self.current_samples_next_offset[xi] == self.current_samples_max_partition_size[xi]:
                return

            x = self.prev_batch_sample_indices[xi]

            match_mask = (batch_sample_indices == x)
            if not np.any(match_mask):
                return

            ni = (np.arange(len(match_mask))[match_mask])[0]
            next_samples_next_offset[ni] = self.current_samples_next_offset[xi]
            next_sample_slices_inds[ni] = self.current_sample_slice_inds[xi]
            next_samples_partitions_starts[ni] = self.current_samples_partitions_starts[xi]
            next_samples_partitions_ends[ni] = self.current_samples_partitions_ends[xi]
            next_samples_max_partition_size[ni] = self.current_samples_max_partition_size[xi]

        v_replace_remaining_samples_info = np.vectorize(replace_remaining_samples_info)
        if len(self.prev_batch_sample_indices) > 0:
            v_replace_remaining_samples_info(np.arange(len(self.prev_batch_sample_indices)))

        self.current_samples_next_offset = next_samples_next_offset
        self.current_sample_slice_inds = next_sample_slices_inds
        self.current_samples_partitions_starts = next_samples_partitions_starts
        self.current_samples_partitions_ends = next_samples_partitions_ends
        self.current_samples_max_partition_size = next_samples_max_partition_size

        def choose_elements_for_sample(index):

            sample_index = batch_sample_indices[index]

            # checking if sample is new, calculating its properties!
            if self.current_samples_partitions_starts[index] is None:

                sample_slice_inds = np.arange(len(
                    self.content_loader_of_interest.sample_slices_left_lobe_indices[sample_index]))

                if len(sample_slice_inds) < elements_per_batch:
                    if len(sample_slice_inds) > 4:
                        sample_slice_inds = np.concatenate((
                            sample_slice_inds[0:2],
                            np.repeat(
                                sample_slice_inds[2:-2],
                                int(np.ceil((elements_per_batch - 4) / (len(sample_slice_inds) - 4)))),
                            sample_slice_inds[-2:]
                        ), axis=0)
                    else:
                        sample_slice_inds = np.repeat(
                                sample_slice_inds,
                                int(np.ceil(elements_per_batch / len(sample_slice_inds))))

                partitions_starts = np.round(
                    np.arange(elements_per_batch) * 1.0 * len(sample_slice_inds) / elements_per_batch).\
                    astype(int)

                partition_ends = np.concatenate((
                    partitions_starts[1:], np.asarray([len(sample_slice_inds)])), axis=0)

                partition_sizes = partition_ends - partitions_starts

                max_ps = np.amax(partition_sizes)

                self.current_samples_partitions_starts[index] = partitions_starts
                self.current_samples_partitions_ends[index] = partition_ends
                self.current_samples_max_partition_size[index] = max_ps
                self.current_sample_slice_inds[index] = sample_slice_inds

            else:
                partitions_starts = self.current_samples_partitions_starts[index]
                partition_ends = self.current_samples_partitions_ends[index]
                sample_slice_inds = self.current_sample_slice_inds[index]
                max_ps = self.current_samples_max_partition_size[index]

            offset = self.current_samples_next_offset[index]
            els = np.minimum(partitions_starts + offset, partition_ends - 1)
            els = sample_slice_inds[els]
            self.current_samples_next_offset[index] += 1

            # now concatenating the related left and right indices!
            els = np.concatenate((
                self.content_loader_of_interest.sample_slices_left_lobe_indices[sample_index][els],
                self.content_loader_of_interest.sample_slices_right_lobe_indices[sample_index][els],
            ), axis=0)

            return els

        ret_val = np.stack(tuple([choose_elements_for_sample(i) for i in range(len(batch_sample_indices))]), axis=0)

        return ret_val

    def reset(self):
        super(SequentialPreciseEquallySpacedSliceChooser, self).reset()
        self.current_samples_next_offset = []
        self.current_samples_partitions_starts = []
        self.current_samples_partitions_ends = []
        self.current_samples_max_partition_size = []
