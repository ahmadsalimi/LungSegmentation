import numpy as np
from Auxiliary.ModelEvaluation.Evaluator import Evaluator


class BinaryEvaluator(Evaluator):

    def __init__(self, model, conf, data_loader):
        super(BinaryEvaluator, self).__init__(model, conf, data_loader)

        self.ground_truth = np.zeros((data_loader.get_number_of_samples(),), dtype=int)
        self.model_positive_class_probs = np.zeros((data_loader.get_number_of_samples(),), dtype=float)
        self.n_repeats = np.zeros((data_loader.get_number_of_samples(),), dtype=float)
        self.n_positive_repeats = np.zeros((data_loader.get_number_of_samples(),), dtype=float)

    def calculate_evaluation_requirements(self, ground_truth, prediction):
        """ Receives as input 2 numpy arrays, one containing he ground truth labels
         and the other containing the labels predicted by the model.
        Returns the number of [TP, TN, FP, FN] in one numpy array."""
        tp = int(np.sum(np.logical_and(ground_truth == 1, prediction == 1)))
        tn = int(np.sum(np.logical_and(ground_truth == 0, prediction == 0)))
        fp = int(np.sum(np.logical_and(ground_truth == 0, prediction == 1)))
        fn = int(ground_truth.shape[0] - tp - tn - fp)

        return np.asarray([tp, tn, fp, fn])

    def aggregate_evaluation_requirements(self,
                                          aggregated_evaluation_requirements, new_evaluation_requirements):
        """ Receives as input 2 numpy arrays of length 4 containing [TP, TN, FP, FN],
        one for the aggregated values and another for the new values to aggregate.
        Returns the new aggregated values. """
        return aggregated_evaluation_requirements + new_evaluation_requirements

    def get_the_number_of_evaluation_requirements(self):
        """ Returns the number of stats that are needed to be kept to calculate the final
        evaluation metrics. """
        return 4

    def calculate_evaluation_metrics(self, aggregated_evaluation_requirements):
        """ Receives as input a numpy array of length 4 containing [TP, TN, FP, FN]
        for the whole data, returns an array containing accuracy, sensitivity and specificity respectively.
        Will return -1 for the un-calculable values (e.g. sensitivity without positive samples)"""

        if self.conf['phase'] != 'train':
            print('### TP, TN, FP, FN ', aggregated_evaluation_requirements)

        tp, tn, fp, fn = tuple(aggregated_evaluation_requirements.tolist())

        p = tp + fn
        n = tn + fp

        if p + n > 0:
            accuracy = (tp + tn) * 100.0 / (n + p)
        else:
            accuracy = -1

        if p > 0:
            sensitivity = 100.0 * tp / p
        else:
            sensitivity = -1

        if n > 0:
            specificity = 100.0 * tn / max(n, 1)
        else:
            specificity = -1

        return np.asarray([accuracy, sensitivity, specificity])

    def get_headers_of_evaluation_metrics(self):
        """ Returns a list containing the titles of metrics respectively"""
        return ['Acc', 'Sens', 'Spec']

    def reset(self):
        self.ground_truth = np.zeros((self.data_loader.get_number_of_samples(),), dtype=int)
        self.model_positive_class_probs = np.zeros((self.data_loader.get_number_of_samples(),), dtype=float)

    def update_variables_based_on_the_output_of_the_model(self, model_output):
        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        self.ground_truth[current_batch_sample_indices] = \
            self.data_loader.get_samples_labels()[current_batch_sample_indices]

        positive_class_probs = model_output['positive_class_probability'].cpu().numpy()

        self.n_repeats[current_batch_sample_indices] += 1
        self.n_positive_repeats[current_batch_sample_indices] += \
            (self.model_positive_class_probs[current_batch_sample_indices] >= 0.5).astype(int)

        self.model_positive_class_probs[current_batch_sample_indices] = \
            np.maximum(positive_class_probs, self.model_positive_class_probs[current_batch_sample_indices])

    def extract_ground_truth_and_predictions_for_batch(self, model_output):
        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        gt = self.data_loader.get_samples_labels()[current_batch_sample_indices]

        positive_class_probs = model_output['positive_class_probability'].cpu().numpy()

        return gt, (positive_class_probs >= 0.5).astype(int)

    def get_ground_truth_and_predictions_for_all_samples(self):
        return self.ground_truth, (self.model_positive_class_probs >= 0.5).astype(int)

    def get_samples_results_summaries(self):
        samples_names = self.data_loader.get_samples_names()
        return samples_names, ['%d\t%.2f\t%d\t%d\t%.2f' %
                               (self.ground_truth[i], self.model_positive_class_probs[i],
                                self.n_repeats[i], self.n_positive_repeats[i],
                                100.0 * self.n_positive_repeats[i] / self.n_repeats[i])
                               for i in range(len(self.model_positive_class_probs))]

    def get_samples_results_header(self):
        return ['GT', 'PositiveClassProbability', 'n_repeat', 'n_positive_repeat', 'positive_ratio']

    def get_samples_elements_results_summaries(self):
        raise Exception('Not applicable to this class')
