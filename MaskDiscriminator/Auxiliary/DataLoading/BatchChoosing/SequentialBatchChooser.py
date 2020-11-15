from Auxiliary.DataLoading.BatchChoosing.BatchChooser import BatchChooser
import numpy as np


class SequentialBatchChooser(BatchChooser):
    """ Returns a balanced batch for all class types,
    each sample of one class has equal probability to be chosen"""

    def __init__(self, config, data_loader, batch_element_chooser=None):
        super(SequentialBatchChooser, self).__init__(config, data_loader, batch_element_chooser)
        self.cursor = 0

    def reset(self):
        super(SequentialBatchChooser, self).reset()
        self.cursor = 0

    def get_next_batch_sample_indices(self, sample_completion_flags):
        """ Returns a list containing the indices of the samples chosen for the next batch."""

        next_sample_inds = self.current_batch_sample_indices[np.logical_not(sample_completion_flags)]

        # if there are less samples than specified by batch, appending new samples
        if len(next_sample_inds) < self.config['batch_size'] and \
                self.cursor < self.dataLoader.get_number_of_samples():
            new_cursor = min(
                self.dataLoader.get_number_of_samples(),
                self.cursor + self.config['batch_size'] - len(next_sample_inds))

            if len(next_sample_inds) > 0 and self.cursor < new_cursor:
                next_sample_inds = np.concatenate((next_sample_inds, np.arange(self.cursor, new_cursor)), axis=0)
            elif len(next_sample_inds) == 0:
                next_sample_inds = np.arange(self.cursor, new_cursor)

            self.cursor = new_cursor

        return next_sample_inds
