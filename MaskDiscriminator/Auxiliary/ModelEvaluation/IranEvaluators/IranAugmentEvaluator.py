from Auxiliary.ModelEvaluation.BinaryEvaluator import BinaryEvaluator
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
import numpy as np
from os import path, makedirs


class IranAugmentEvaluator(BinaryEvaluator):

    def __init__(self, model, conf, data_loader):
        super(BinaryEvaluator, self).__init__(model, conf, data_loader)

        self.ground_truth = np.zeros((data_loader.get_number_of_samples(),), dtype=int)
        self.model_positive_class_probs = np.zeros((data_loader.get_number_of_samples(),), dtype=float)
        self.n_repeats = np.zeros((data_loader.get_number_of_samples(),), dtype=float)
        self.n_positive_repeats = np.zeros((data_loader.get_number_of_samples(),), dtype=float)

        self.table_loader = data_loader.content_loaders[0]
        self.current_batch = None
        self.original_ids = None
        self.augment_results = None
        self.augment_gt = None
        self.model_loss_max_augment = np.zeros((data_loader.get_number_of_samples(),), dtype=float)

        self.indices = np.array([])

        self.whole_worse_case_samples = 0

    def calculate_evaluation_requirements(self, ground_truth_, prediction_):
        """ Receives as input 2 numpy arrays, one containing he ground truth labels
         and the other containing the labels predicted by the model.
        Returns the number of [TP, TN, FP, FN] in one numpy array."""
        if self.conf['phase'] == 'train':
            b = int(prediction_.shape[0] / self.conf['repeat_value'])
            prediction = prediction_.copy()[[i * self.conf['repeat_value'] for i in range(0, b)]]
            ground_truth = ground_truth_[[i * self.conf['repeat_value'] for i in range(0, b)]]
            tp = int(np.sum(np.logical_and(ground_truth == 1, prediction == 1)))
            tn = int(np.sum(np.logical_and(ground_truth == 0, prediction == 0)))
            fp = int(np.sum(np.logical_and(ground_truth == 0, prediction == 1)))
            fn = int(ground_truth.shape[0] - tp - tn - fp)
            prediction = prediction_.copy().reshape((b, self.conf['repeat_value']))
        elif self.conf['phase'] == 'eval':
            prediction = prediction_.copy()
            ground_truth = ground_truth_.copy()
            tp = int(np.sum(np.logical_and(ground_truth == 1, prediction == 1)))
            tn = int(np.sum(np.logical_and(ground_truth == 0, prediction == 0)))
            fp = int(np.sum(np.logical_and(ground_truth == 0, prediction == 1)))
            fn = int(ground_truth.shape[0] - tp - tn - fp)
            prediction = self.augment_results.copy()
            ground_truth = self.augment_gt.copy()

        differences_dead = prediction[ground_truth == 1] - np.repeat(
            prediction[ground_truth == 1, 0],
            prediction[ground_truth == 1, :].shape[1]).reshape(
            prediction[ground_truth == 1, :].shape)
        differences_dead[differences_dead > 0] = 0
        # differences_dead = np.sum(differences_dead, axis=0)
        differences_dead[differences_dead < 0] = 1
        # differences_dead[differences_dead > 0] = 0
        worse_cases_dead = (4 * ground_truth.shape[0]) / np.sum(differences_dead)

        # ee = self.indices[ground_truth == 1]
        # print(ee[differences_dead > 0])
        del differences_dead

        differences_alive = prediction[ground_truth == 0] - np.repeat(
            prediction[ground_truth == 0, 0],
            prediction[ground_truth == 0, :].shape[1]).reshape(
            prediction[ground_truth == 0, :].shape)
        differences_alive[differences_alive < 0] = 0
        # differences_alive = np.sum(differences_alive, axis=1)
        differences_alive[differences_alive > 0] = 1
        # differences_alive[differences_alive < 0] = 0
        worse_cases_alive = (4 * ground_truth.shape[0]) / np.sum(differences_alive)

        # ee = self.indices[ground_truth == 0]
        # print(ee[differences_alive > 0])
        del differences_alive

        label_augmented = (prediction >= 0.5).astype(int)
        label_differences_dead = label_augmented[ground_truth == 1] - np.repeat(
            label_augmented[ground_truth == 1, 0],
            label_augmented[ground_truth == 1, :].shape[1]).reshape(
            label_augmented[ground_truth == 1, :].shape)
        label_differences_dead[label_differences_dead > 0] = 0
        # label_differences_dead = np.sum(label_differences_dead, axis=1)
        label_differences_dead[label_differences_dead < 0] = 1
        # label_differences_dead[label_differences_dead > 0] = 0
        worse_cases_dead_label = (4 * ground_truth.shape[0]) / np.sum(label_differences_dead)

        # ee = self.indices[ground_truth == 1]
        # print(ee[label_differences_dead > 0])
        del label_differences_dead

        label_differences_alive = label_augmented[ground_truth == 0] - np.repeat(
            label_augmented[ground_truth == 0, 0],
            label_augmented[ground_truth == 0, :].shape[1]).reshape(
            label_augmented[ground_truth == 0, :].shape)
        label_differences_alive[label_differences_alive < 0] = 0
        # label_differences_alive = np.sum(label_differences_alive, axis=1)
        label_differences_alive[label_differences_alive > 0] = 1
        # label_differences_alive[label_differences_alive < 0] = 0
        worse_cases_alive_label = (4 * ground_truth.shape[0]) / np.sum(label_differences_alive)

        # ee = self.indices[ground_truth == 0]
        # print(ee[label_differences_alive > 0])
        del label_differences_alive
        return np.asarray([tp, tn, fp, fn, worse_cases_dead, worse_cases_alive, worse_cases_dead_label,
                           worse_cases_alive_label])

    def aggregate_evaluation_requirements(self,
                                          aggregated_evaluation_requirements, new_evaluation_requirements):
        """ Receives as input 2 numpy arrays of length 4 containing [TP, TN, FP, FN],
        one for the aggregated values and another for the new values to aggregate.
        Returns the new aggregated values. """
        return aggregated_evaluation_requirements + new_evaluation_requirements

    def get_the_number_of_evaluation_requirements(self):
        """ Returns the number of stats that are needed to be kept to calculate the final
        evaluation metrics. """
        return 8

    def calculate_evaluation_metrics(self, aggregated_evaluation_requirements):
        """ Receives as input a numpy array of length 4 containing [TP, TN, FP, FN]
        for the whole data, returns an array containing accuracy, sensitivity and specificity respectively.
        Will return -1 for the un-calculable values (e.g. sensitivity without positive samples)"""

        if self.conf['phase'] != 'train':
            print(
                '### TP, TN, FP, FN, worse_cases_dead, worse_cases_alive, worse_cases_dead_label, '
                'worse_cases_alive_label',
                aggregated_evaluation_requirements)

        tp, tn, fp, fn, worse_cases_dead, worse_cases_alive, worse_cases_dead_label, worse_cases_alive_label = tuple(
            aggregated_evaluation_requirements.tolist())

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
        return np.asarray(
            [accuracy, sensitivity, specificity, worse_cases_dead, worse_cases_alive, worse_cases_dead_label,
             worse_cases_alive_label])

    def get_headers_of_evaluation_metrics(self):
        """ Returns a list containing the titles of metrics respectively"""
        return ['Acc', 'Sens', 'Spec', 'worse_cases_dead', 'worse_cases_alive', 'worse_cases_dead_label',
                'worse_cases_alive_label']

    def reset(self):
        self.ground_truth = np.zeros((self.data_loader.get_number_of_samples(),), dtype=int)
        self.model_positive_class_probs = np.zeros((self.data_loader.get_number_of_samples(),), dtype=float)

    def update_variables_based_on_the_output_of_the_model(self, model_output):
        self.current_batch = self.table_loader.new_batch
        # self.original_ids = self.table_loader.original_indices
        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        self.ground_truth[current_batch_sample_indices] = \
            self.data_loader.get_samples_labels()[current_batch_sample_indices]
        positive_class_probability = model_output['positive_class_probability'].cpu().numpy()
        b = int(positive_class_probability.shape[0] / self.conf['repeat_value'])
        positive_class_probs = positive_class_probability[
            [i * self.conf['repeat_value'] for i in range(0, b)]]
        # positive_class_probs = positive_class_probability
        gt, _ = self.extract_ground_truth_and_predictions_for_batch(model_output)
        gt = gt[[i * self.conf['repeat_value'] for i in range(0, b)]]
        if self.augment_results is None:
            self.augment_results = positive_class_probability.reshape(-1, 4).copy()
            self.augment_gt = gt
            self.indices = current_batch_sample_indices
        else:
            self.augment_results = np.concatenate((self.augment_results, positive_class_probability.reshape(-1, 4)), axis=0)
            self.augment_gt = np.concatenate((self.augment_gt, gt), axis=0)
            self.indices += current_batch_sample_indices
        self.n_repeats[current_batch_sample_indices] += 1
        self.n_positive_repeats[current_batch_sample_indices] += \
            (self.model_positive_class_probs[current_batch_sample_indices] >= 0.5).astype(int)
        # self.n_positive_repeats[current_batch_sample_indices] += \
        #     self.model_positive_class_probs[current_batch_sample_indices]

        self.model_positive_class_probs[current_batch_sample_indices] = \
            np.maximum(positive_class_probs, self.model_positive_class_probs[current_batch_sample_indices])

    def extract_ground_truth_and_predictions_for_batch(self, model_output):
        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        gt = self.data_loader.get_samples_labels()[
            np.repeat(current_batch_sample_indices, self.conf['repeat_value'], axis=0)]

        positive_class_probs = model_output['positive_class_probability'].cpu().numpy()
        return gt, (positive_class_probs >= 0.5).astype(int)

    def get_ground_truth_and_predictions_for_all_samples(self):
        return self.ground_truth, (self.model_positive_class_probs >= 0.5).astype(int)

    def get_samples_results_summaries(self):
        samples_names = self.data_loader.get_samples_names()
        return samples_names, ['%.2f\t%d\t%d\t%.2f' %
                               (self.model_positive_class_probs[i], self.n_repeats[i], self.n_positive_repeats[i],
                                100.0 * self.n_positive_repeats[i] / self.n_repeats[i])
                               for i in range(len(self.model_positive_class_probs))]

    def get_samples_results_header(self):
        return ['PositiveClassProbability', 'n_repeat', 'n_positive_repeat', 'positive_ratio']

    def get_samples_elements_results_summaries(self):
        raise Exception('Not applicable to this class')

    def save_output_of_the_model_for_batch(self, model_output):
        pass
