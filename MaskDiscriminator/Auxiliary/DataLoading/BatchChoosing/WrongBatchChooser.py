from Auxiliary.DataLoading.BatchChoosing.BatchChooser import BatchChooser
import numpy as np


class WrongBatchChooser(BatchChooser):
    """ Returns a balanced batch for all class types,
    each sample of one class has equal probability to be chosen"""

    def __init__(self, config, data_loader, batch_element_chooser=None,
                 wrong_samples_choosing_prob=0.8):
        super(WrongBatchChooser, self).__init__(config, data_loader, batch_element_chooser)

        # selecting the size of batch <= the given batch size which is a multiplier of the number of classes
        self.new_batch_size = self.config['batch_size'] - \
                              (self.config['batch_size'] % len(self.class_samples_indices))
        if self.new_batch_size != self.config['batch_size']:
            print(
                'Continuing with batch size of %d which is a multiplier of the number of classes.' % self.new_batch_size)

        self.wrong_samples_choosing_prob = wrong_samples_choosing_prob

    def get_next_batch_sample_indices(self, sample_completion_flags):
        """ Returns a list containing the indices of the samples chosen for the next batch."""

        #print('\n#######')

        def choose_samples_for_class(the_class):
            sample_inds = self.class_samples_indices[the_class]
            samples_wrong_mask = (
                    self.dataLoader.samples_labels[sample_inds] !=
                    (self.dataLoader.model_preds_for_samples[sample_inds] >= 0.5).astype(int))

            if np.all(samples_wrong_mask):
                print('Warning: all wrong in class ', the_class)
            elif not np.any(samples_wrong_mask):
                print('Warning: no mistakes in class ', the_class)

            # a binomial test to choose n samples from mistakes and the rest from all
            mistakes_cnt = np.random.binomial(n_samples_per_class, self.wrong_samples_choosing_prob, 1)
            if mistakes_cnt > 0 and np.any(samples_wrong_mask):
                mistakes_inds = sample_inds[samples_wrong_mask]
                mistakes_inds = mistakes_inds[
                    np.random.randint(0, len(mistakes_inds), mistakes_cnt)]
            else:
                mistakes_inds = np.asarray([])

            final_choices = np.concatenate((
                mistakes_inds,
                sample_inds[np.random.randint(0, len(sample_inds), n_samples_per_class - len(mistakes_inds))]
            ), axis=0).astype(int)

            #print('Chosen batch for the class ', the_class)
            #print('\t', final_choices)
            #print('\tGT:', self.dataLoader.samples_labels[final_choices])
            #print('\tPred:', self.dataLoader.model_preds_for_samples[final_choices])

            return final_choices

        n_samples_per_class = int(self.new_batch_size / len(self.class_samples_indices))

        class_indices = sorted(list(self.class_samples_indices.keys()))

        # choosing samples for each class and concatenating them
        chosen_sample_indices = np.stack(tuple([
            choose_samples_for_class(c)
            for c in class_indices
        ]), axis=1)

        # Checking out the kept samples
        kept_samples = self.current_batch_sample_indices[np.logical_not(sample_completion_flags)]
        kept_samples_labels = self.dataLoader.get_samples_labels()[kept_samples]
        sample_cursors_per_class = np.zeros((len(class_indices),), dtype=int)
        for i in range(len(kept_samples)):
            l = kept_samples_labels[i]
            chosen_sample_indices[sample_cursors_per_class[l], kept_samples_labels[i]] = kept_samples[i]
            sample_cursors_per_class[l] += 1

        # (For shuffling, so we would have 0 1 0 1 0 1 instead of 0 0 0 1 1 1 which is required for
        # having even batches in running the model in multiple GPUS)

        chosen_sample_indices = chosen_sample_indices.reshape(-1)

        return chosen_sample_indices

