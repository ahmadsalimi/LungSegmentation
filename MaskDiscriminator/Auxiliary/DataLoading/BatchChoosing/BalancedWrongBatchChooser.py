from Auxiliary.DataLoading.BatchChoosing.BatchChooser import BatchChooser
import numpy as np


class BalancedWrongBatchChooser(BatchChooser):

    def __init__(self, config, data_loader, batch_element_chooser=None, wrong_samples_choosing_prob=0.3):
        super(BalancedWrongBatchChooser, self).__init__(config, data_loader, batch_element_chooser)

        self.samples_labels = self.dataLoader.get_samples_labels()
        self.samples_paths = self.dataLoader.get_samples_names()

        self.class_hospitals_sample_indices = self.separate_samples_of_hospitals_in_classes()
        self.class_samples_indices = self.dataLoader.get_class_sample_indices()

        # selecting the size of batch <= the given batch size which is a multiplier of the number of classes
        self.new_batch_size = self.config['batch_size'] - \
                              (self.config['batch_size'] % len(self.class_samples_indices))

        if self.new_batch_size != self.config['batch_size']:
            print(
                'Continuing with batch size of %d which is a multiplier of the number of classes.' % self.new_batch_size)

        self.wrong_samples_choosing_prob = wrong_samples_choosing_prob

    def separate_samples_of_hospitals_in_classes(self):
        """ Finds the index of samples belonging to different hospitals in different classes and fills
        class_hospitals_sample_indices dictionary. """

        def find_hospital(sample_path):
            """ Finds the hospital name from the path of the sample and returns it.
            One directory before each sample is its group/hospital. """
            if sample_path.endswith('/'):
                sample_path = sample_path[:-1]
            return sample_path[:sample_path.rfind('/')]

        class_hospitals_dict = dict()
        for i in range(len(self.samples_labels)):

            # Initiating a dictionary for each class if has not been added before
            if self.samples_labels[i] not in class_hospitals_dict:
                class_hospitals_dict[self.samples_labels[i]] = dict()

            hospital_dict = class_hospitals_dict[self.samples_labels[i]]
            hospital_name = find_hospital(self.samples_paths[i])

            # Initiating a list for each hospital, if not done before
            if hospital_name not in hospital_dict:
                hospital_dict[hospital_name] = []

            # Adding the sample index to the dictionary of the hospital
            hospital_dict[hospital_name].append(i)

        return class_hospitals_dict

    def get_next_batch_sample_indices(self, sample_completion_flags):
        """ Returns a list containing the indices of the samples chosen for the next batch."""

        n_samples_per_class = int(self.new_batch_size / len(self.class_samples_indices))

        def sample_per_hospital(hospital_sample_inds):
            """ Returns the index of the random sample chosen from among the list of sample indices."""

            # check if a wrong sample should be chosen
            wrong_sample = np.random.binomial(1, self.wrong_samples_choosing_prob, 1)
            if wrong_sample[0] == 1:
                samples_wrong_mask = (
                        self.dataLoader.samples_labels[hospital_sample_inds] !=
                        (self.dataLoader.model_preds_for_samples[hospital_sample_inds] >= 0.5).astype(int))
                wrong_inds = np.asarray(hospital_sample_inds)[samples_wrong_mask]
                if np.sum(samples_wrong_mask) > 0:
                    return wrong_inds[np.random.randint(0, len(wrong_inds))]

            ri = np.random.randint(0, len(hospital_sample_inds))
            return hospital_sample_inds[ri]

        def sample_for_each_class(class_label):
            """ Returns the indices of random samples chosen for the given class label """
            class_hospitals = list(self.class_hospitals_sample_indices[class_label].keys())
            rand_hospitals = np.random.randint(
                0, len(class_hospitals), size=(n_samples_per_class,))

            rand_indices = [
                sample_per_hospital(self.class_hospitals_sample_indices[class_label][class_hospitals[h]])
                for h in rand_hospitals]

            return np.asarray(rand_indices)

        # choosing samples for each class and concatenating them
        # (For shuffling, so we would have 0 1 0 1 0 1 instead of 0 0 0 1 1 1 which is required for
        # having even batches in running the model in multiple GPUS)
        chosen_sample_indices = np.stack(tuple([
            sample_for_each_class(c)
            for c in self.class_samples_indices.keys()
        ]), axis=1)

        chosen_sample_indices = chosen_sample_indices.reshape(-1)
        return chosen_sample_indices
