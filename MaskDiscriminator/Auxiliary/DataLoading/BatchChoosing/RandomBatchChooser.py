from Auxiliary.DataLoading.BatchChoosing.BatchChooser import BatchChooser
import numpy as np


class RandomBatchChooser(BatchChooser):
    """ Returns a balanced batch for all class types,
    each sample of one class has equal probability to be chosen"""

    def __init__(self, config, data_loader, batch_element_chooser=None):
        super(RandomBatchChooser, self).__init__(config, data_loader, batch_element_chooser)

        self.classes = sorted(list(self.class_samples_indices.keys()))
        self.class_index_dict = dict(zip(self.classes, np.arange(len(self.classes))))
        self.index_class_dict = dict(zip(np.arange(len(self.classes)), self.classes))
        self.class_extra_samples_added = np.zeros(len(self.classes), dtype=int)

    def get_next_batch_sample_indices(self, sample_completion_flags):
        """ Returns a list containing the indices of the samples chosen for the next batch."""

        kept_samples = self.current_batch_sample_indices[np.logical_not(sample_completion_flags)]

        if len(kept_samples) == self.config['batch_size']:
            return kept_samples

        if len(kept_samples) > 0:
            kept_samples_labels = self.dataLoader.get_samples_labels()[kept_samples]
            np.add.at(
                self.class_extra_samples_added,
                np.vectorize(lambda x: self.class_index_dict[x])(kept_samples_labels),
                1)
        else:
            kept_samples = np.zeros((0,), dtype=int)

        # deciding for the classes that samples of the batch should be chosen from them
        while len(kept_samples) < self.config['batch_size']:
            zero_debt_classes_inds = \
                np.arange(len(self.classes))[self.class_extra_samples_added == 0]

            if len(zero_debt_classes_inds) <= self.config['batch_size'] - len(kept_samples):
                reduce_extra = True
            else:
                reduce_extra = False

            zero_debt_classes_inds = zero_debt_classes_inds[:
                                                            min(
                                                                self.config['batch_size'] - len(kept_samples),
                                                                len(zero_debt_classes_inds)
                                                            )]

            self.class_extra_samples_added[zero_debt_classes_inds] += 1

            kept_samples = np.concatenate((
                kept_samples,
                np.asarray([
                    self.class_samples_indices[self.index_class_dict[ci]][
                        np.random.randint(
                            0,
                            len(self.class_samples_indices[self.index_class_dict[ci]]),
                            1)[0]]
                    for ci in zero_debt_classes_inds
                ])), axis=0)

            if reduce_extra:
                self.class_extra_samples_added -= 1

        return kept_samples
