import numpy as np
from Auxiliary.ModelEvaluation.Evaluator import Evaluator
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader


class TertiaryCovidEvaluator(Evaluator):
    """ The model should have class1_elements_probs and class2_elements_probs in its return dict"""

    def __init__(self, model, conf, data_loader):
        super(TertiaryCovidEvaluator, self).__init__(model, conf, data_loader)

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

        self.ground_truth = np.zeros((self.data_loader.get_number_of_samples(),), dtype=int)

        self.sample_elements_coro_probs = \
            [np.zeros((len(elements_names),), dtype=float) for elements_names in
             self.content_loader_of_interest.get_elements_names()]
        self.sample_elements_abn_probs = \
            [np.zeros((len(elements_names),), dtype=float) for elements_names in
             self.content_loader_of_interest.get_elements_names()]

    def reset(self):
        self.ground_truth = np.zeros((self.data_loader.get_number_of_samples(),), dtype=int)

        self.sample_elements_coro_probs = \
            [np.zeros((len(elements_names),), dtype=float) for elements_names in
             self.content_loader_of_interest.get_elements_names()]
        self.sample_elements_abn_probs = \
            [np.zeros((len(elements_names),), dtype=float) for elements_names in
             self.content_loader_of_interest.get_elements_names()]

    def calculate_evaluation_requirements(self, ground_truth, prediction):
        """ Receives as input ground truth which is a numpy array containing the label
         of the samples and a tuple of 2 numpy arrays, one containing the predicted label for corona,
         and the other containing the predicted label for abnormal.
        Returns fattened confusion matrix (3 x 3 -> 9)."""

        coro_label, abn_label = prediction

        # if corona is on, no matter abnormal is 1 or 0, the sample will be considered as a corona sample
        final_label = coro_label + (1 - coro_label) * 2 * abn_label

        # confusion matrix
        conf_count = np.asarray([[np.sum(np.logical_and(ground_truth == r, final_label == c))
                                  for c in range(3)]
                                 for r in range(3)])

        return conf_count.flatten()

    def aggregate_evaluation_requirements(self,
                                          aggregated_evaluation_requirements, new_evaluation_requirements):
        """ Receives as input 2 numpy arrays of length 9 containing flattened confusion matrices,
        one for the aggregated values and another for the new values to aggregate.
        Returns the new aggregated values. """
        return aggregated_evaluation_requirements + new_evaluation_requirements

    def get_the_number_of_evaluation_requirements(self):
        """ Returns the number of stats that are needed to be kept to calculate the final
        evaluation metrics. """
        return 9

    def calculate_evaluation_metrics(self, aggregated_evaluation_requirements):
        """ Receives as input a numpy array of length 9 containing flattened confusion matrix
        for the whole data, returns an array containing accuracy,
        disease sensitivity (Corona + Abnormal vs Healthy), Specificity (Abnormal + healthy vs corona),
        and sensitivity per each class (for healthy, corona, abnormal classes) respectively.
        Will return -1 for the un-calculable values (e.g. sensitivity without positive samples)"""

        conf_count = aggregated_evaluation_requirements.reshape((3, 3))

        if self.conf['phase'] != 'train':
            print('### confusion matrix (H C A):')
            print('\n'.join(['\t'.join(['%d' % conf_count[r, c] for c in range(3)]) for r in range(3)]))
            print('###')

        n_classes = np.asarray([np.sum(conf_count[i, :]) for i in range(3)])

        def get_sens(i):
            """ Calculates sensitivity for class i"""
            n_i = n_classes[i]
            if n_i == 0:
                return -1
            else:
                return 100.0 * conf_count[i, i] / n_i

        if np.sum(n_classes) == 0:
            acc = -1
        else:
            acc = 100.0 * np.sum([conf_count[i, i] for i in range(3)]) / np.sum(n_classes)

        if np.sum(n_classes[[1, 2]]) == 0:
            disease_sens = -1
        else:
            disease_sens = 100.0 * sum([conf_count[i, j] for i in [1, 2] for j in [1, 2]]) / np.sum(n_classes[[1, 2]])

        if np.sum(n_classes[[0, 2]]) == 0:
            spec = -1
        else:
            spec = 100.0 * sum([conf_count[i, j] for i in [0, 2] for j in [0, 2]]) / np.sum(n_classes[[0, 2]])

        return acc, disease_sens, spec, get_sens(0), get_sens(1), get_sens(2)

    def get_headers_of_evaluation_metrics(self):
        """ Returns a list containing the titles of metrics respectively,
        Accuracy, Disease sensitivity (A + C vs H), Noncorona Specificity (A + H vs C),
        Healthy, Corona and Abnormal sensitivities."""
        return ['Acc', 'DSens', 'NCSpec', 'HSens', 'CSens', 'ASens']

    def update_variables_based_on_the_output_of_the_model(self, model_output):

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_elements_indices = self.data_loader.get_current_batch_elements_indices()

        self.ground_truth[current_batch_sample_indices] = \
            self.data_loader.get_samples_labels()[current_batch_sample_indices]

        slice_coro_probs = model_output['class1_elements_probs'].cpu().numpy()
        slice_abn_probs = model_output['class2_elements_probs'].cpu().numpy()

        # updating probability of slices
        for bsi in range(len(current_batch_sample_indices)):
            # mask of the elements in range
            ok_mask = (current_batch_elements_indices[bsi] >= 0)
            (self.sample_elements_coro_probs[
                current_batch_sample_indices[bsi]][current_batch_elements_indices[bsi][ok_mask]]) = \
                (slice_coro_probs[bsi, :])[ok_mask]
            (self.sample_elements_abn_probs[
                current_batch_sample_indices[bsi]][current_batch_elements_indices[bsi][ok_mask]]) = \
                (slice_abn_probs[bsi, :])[ok_mask]

    def extract_ground_truth_and_predictions_for_batch(self, model_output):

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_elements_indices = self.data_loader.get_current_batch_elements_indices()

        gt = self.data_loader.get_samples_labels()[current_batch_sample_indices]

        slice_coro_probs = model_output['class1_elements_probs'].cpu().numpy()
        slice_abn_probs = model_output['class2_elements_probs'].cpu().numpy()

        samples_coro_probs = np.asarray([
            np.amax((slice_coro_probs[i, :])[current_batch_elements_indices[i, :] > -1])
            for i in range(len(current_batch_sample_indices))])

        samples_abn_probs = np.asarray([
            np.amax((slice_abn_probs[i, :])[current_batch_elements_indices[i, :] > -1])
            for i in range(len(current_batch_sample_indices))])

        return gt, (samples_coro_probs >= 0.5).astype(int), (samples_abn_probs >= 0.5).astype(int)

    def decide_for_binary_classification(self, slice_paths, class_x_slice_probs):
        """ Receives as input the list of slice paths and an array containing probabilities of slices
        for class X for one sample. Calculates the probability of sample belonging to class X
        based on their softened slice probabilities for class X."""
        
        def soften_probs(snames, sprobs):
            left_mask = np.asarray(['left' in sn for sn in snames])
            left_probs = sprobs[left_mask]
            right_probs = sprobs[np.logical_not(left_mask)]

            # softening
            def soften(arr, ind):
                neighbs = []
                if ind > 0:
                    neighbs.append(arr[ind - 1])
                if ind < len(arr) - 1:
                    neighbs.append(arr[ind + 1])

                if len(neighbs) == 0:
                    return arr[ind]

                return min(max(neighbs), arr[ind])

            soft_left = np.vectorize(lambda ind: soften(left_probs, ind))
            soft_right = np.vectorize(lambda ind: soften(right_probs, ind))

            if len(left_probs) > 0:
                softened_left_probs = soft_left(np.arange(len(left_probs)))
            else:
                softened_left_probs = np.asarray([])

            if len(right_probs) > 0:
                softened_right_probs = soft_right(np.arange(len(right_probs)))
            else:
                softened_right_probs = np.asarray([])

            return np.concatenate((softened_left_probs, softened_right_probs), axis=0)

        softened_slices_probs = soften_probs(slice_paths, class_x_slice_probs)
        return np.amax(softened_slices_probs)

    def extract_disease_probs(self):
        """ Returns 2 arrays containing corona probability and abnormal probability for samples"""

        samples_slices_paths = self.content_loader_of_interest.get_elements_names()

        sample_corona_probs = np.zeros((len(samples_slices_paths),))
        sample_abn_probs = np.zeros((len(samples_slices_paths),))

        for samind in range(len(samples_slices_paths)):

            slices_disease_probs = self.sample_elements_coro_probs[samind] + \
                                   self.sample_elements_abn_probs[samind]

            unhealthy_prob = self.decide_for_binary_classification(
                samples_slices_paths[samind], slices_disease_probs)

            coro_prob = self.decide_for_binary_classification(
                samples_slices_paths[samind],
                self.sample_elements_coro_probs[samind] / np.maximum(slices_disease_probs, 1e-36))

            if unhealthy_prob < 0.5:
                cp = unhealthy_prob * coro_prob
                ap = unhealthy_prob * (1 - coro_prob)
            else:
                cp = coro_prob
                ap = 1 - coro_prob

            sample_corona_probs[samind] = cp
            sample_abn_probs[samind] = ap

        return sample_corona_probs, sample_abn_probs

    def get_ground_truth_and_predictions_for_all_samples(self):

        sample_corona_probs, sample_abn_probs = self.extract_disease_probs()

        return self.ground_truth, \
               ((sample_corona_probs >= 0.5).astype(int),
                (sample_abn_probs >= 0.5).astype(int))

    def get_samples_results_summaries(self):

        sample_corona_probs, sample_abn_probs = self.extract_disease_probs()

        return self.data_loader.get_samples_names(), \
               ['%.2f\t%.2f' % (sample_corona_probs[i], sample_abn_probs[i]) for i in range(len(sample_corona_probs))]

    def get_samples_results_header(self):
        """ Returns the header of the results summaries saved for the samples in the report. """
        return ['CoronaProb', 'AbnProb']

    def get_samples_elements_results_summaries(self):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of lists of values (one list per sample in which a value per element).
        """
        raise Exception('Not applicable here.')
        return None
