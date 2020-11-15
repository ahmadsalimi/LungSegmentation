from Auxiliary.DataLoading.BatchChoosing.BatchChooser import BatchChooser
import numpy as np


class NaturalDistrRandomBatchChooser(BatchChooser):

    def __init__(self, config, data_loader, batch_element_chooser=None):
        super(NaturalDistrRandomBatchChooser, self).__init__(config, data_loader, batch_element_chooser)

        self.n_samples = len(self.dataLoader.get_samples_names())

    def get_next_batch_sample_indices(self, sample_completion_flags):
        """ Returns a list containing the indices of the samples chosen for the next batch."""

        if np.any(np.logical_not(sample_completion_flags)):
            remaining_samples = self.current_batch_sample_indices[np.logical_not(sample_completion_flags)]
        else:
            remaining_samples = np.asarray([])

        n_new = self.config['batch_size'] - len(remaining_samples)
        if n_new > 0:
            new_samples = np.random.randint(0, self.n_samples, n_new)

        if len(remaining_samples) == 0:
            return new_samples
        elif n_new == 0:
            return remaining_samples
        else:
            return np.concatenate((remaining_samples, new_samples), axis=0)
