import numpy as np
from Auxiliary.ModelEvaluation.CovidEvaluators.TertiaryCovidEvaluator import TertiaryCovidEvaluator


class OvOEvaluator(TertiaryCovidEvaluator):

    def __init__(self, model, conf):
        """ Model is a predictor, conf is a dictionary containing configurations and settings """
        super(TertiaryCovidEvaluator, self).__init__(model, conf)

    def decide_for_sample(self, cvh, cva, avh):
        """ Decides for sample labels based on the probabilities received by binary classifiers,
        corona vs healthy, corona vs abnormal and abnormal vs healthy"""

        class_pollings = np.zeros((len(cvh), 3), dtype=int)
        class_scores = np.zeros((len(cvh), 3), dtype=float)

        class_scores[:, 0] += (1 - cvh) + (1 - avh)
        class_scores[:, 1] += cvh + cva
        class_scores[:, 2] += avh + (1 - cva)

        class_pollings[:, 0] += (cvh < 0.5).astype(int) + (avh < 0.5).astype(int)
        class_pollings[:, 1] += (cvh >= 0.5).astype(int) + (cva >= 0.5).astype(int)
        class_pollings[:, 2] += (avh >= 0.5).astype(int) + (cva < 0.5).astype(int)

        def decide_for_sample(si):
            # if polling > 1, return argmax, otherwise return max score
            besti = np.argmax(class_pollings[si, :])
            if class_pollings[si, besti] > 1:
                return besti
            else:
                return np.argmax(class_scores[si, :])

        final_label = np.vectorize(decide_for_sample)(np.arange(len(cvh)))
        return final_label

    def calculate_evaluation_requirements(self, ground_truth, prediction):
        """ Receives as input ground truth which is a numpy array containing the label
        of the samples and a tuple of 3 numpy arrays, containing probabilities of
        corona vs healthy, corona vs abnormal and abnormal vs healthy respectively.
        Returns fattened confusion matrix (3 x 3 -> 9)."""

        cvh, cva, avh = prediction
        final_labels = self.decide_for_sample(cvh, cva, avh)

        # confusion matrix
        conf_count = np.asarray([[np.sum(np.logical_and(ground_truth == r, final_labels == c))
                                  for c in range(3)]
                                 for r in range(3)])

        return conf_count.flatten()

    def initiate_required_variables(self, data_loader):
        """ Receives as input the data_loader which contains information about the samples,
        Initiates required variables for keeping the ground truth about the samples required for evaluation,
        required variables (e.g. information about data or predictions of the model). """

        vars_dict = dict()

        # sample level labels
        vars_dict['ground_truth'] = np.zeros((data_loader.get_samples_num(),), dtype=int)

        # slice level labels
        slices_names = data_loader.get_slices_paths()

        # probabilities of binary classifiers, coro vs h, coro vs abn, abn vs h
        vars_dict['slices_cvh_probs'] = [np.zeros((len(ss),)) for ss in slices_names]
        vars_dict['slices_cva_probs'] = [np.zeros((len(ss),)) for ss in slices_names]
        vars_dict['slices_avh_probs'] = [np.zeros((len(ss),)) for ss in slices_names]

        return vars_dict

    def update_variables_based_on_the_output_of_the_model(self, data_loader, dict_of_vars, model_output):
        """ Receives as input a data_loader for loading data and a dictionary containing
        the saved variables. Updates the required variables from the outputs of the model."""

        current_batch_sample_indices, current_batch_slice_indices = data_loader.get_current_batch_indices()

        dict_of_vars['ground_truth'][current_batch_sample_indices] = \
            data_loader.get_sample_labels(current_batch_sample_indices)

        slices_cvh_probs = model_output[0].cpu().numpy()
        slices_cva_probs = model_output[1].cpu().numpy()
        slices_avh_probs = model_output[2].cpu().numpy()

        # updating probability of slices
        for bsi in range(len(current_batch_sample_indices)):
            dict_of_vars['slices_cvh_probs'][current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi]] = \
                slices_cvh_probs[bsi, :]
            dict_of_vars['slices_cva_probs'][current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi]] = \
                slices_cva_probs[bsi, :]
            dict_of_vars['slices_avh_probs'][current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi]] = \
                slices_avh_probs[bsi, :]

    def extract_ground_truth_and_predictions_for_batch(self, data_loader, model_output):
        """ Returns the ground truth and the output of the model
        (a tuple containing corona vs healthy, corona vs abnormal and abnormal vs healthy probabilities)
        related to the current batch which would be used in evaluation. """

        current_batch_sample_indices, _ = data_loader.get_current_batch_indices()
        return data_loader.get_sample_labels(current_batch_sample_indices), \
               (
                   np.amax(model_output[0].cpu().numpy(), axis=1),
                   np.amax(model_output[1].cpu().numpy(), axis=1),
                   np.amax(model_output[2].cpu().numpy(), axis=1)
               )

    def extract_ground_truth_and_predictions(self, data_loader, dict_of_vars):
        """ Reads the ground truth about all of the samples and calculates the probabilities for
        corona vs healthy, corona vs abnormal and abnormal vs healthy binary classifications
        based on the predictions of the model and returns them. Returns an array for ground truth
        and a tuple of three arrays for the mentioned probabilities."""

        ground_truth = dict_of_vars['ground_truth']

        samples_slices_paths = data_loader.get_slices_paths()

        sample_cvh_probs = np.zeros((len(ground_truth),))
        sample_cva_probs = np.zeros((len(ground_truth),))
        sample_avh_probs = np.zeros((len(ground_truth),))

        for samind in range(len(ground_truth)):

            sample_cvh_probs[samind] = self.decide_for_binary_classification(
                samples_slices_paths[samind], dict_of_vars['slices_cvh_probs'][samind])

            sample_cva_probs[samind] = self.decide_for_binary_classification(
                samples_slices_paths[samind], dict_of_vars['slices_cva_probs'][samind])

            sample_avh_probs[samind] = self.decide_for_binary_classification(
                samples_slices_paths[samind], dict_of_vars['slices_avh_probs'][samind])

        return ground_truth, (sample_cvh_probs, sample_cva_probs, sample_avh_probs)

    def get_samples_results_summaries(self, data_loader, dict_of_vars):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of strings (One string per sample containing the summaries related
        to the sample.) """

        _, model_preds = self.extract_ground_truth_and_predictions(data_loader, dict_of_vars)
        cvh_probs, cva_probs, avh_probs = model_preds
        final_labels = self.decide_for_sample(cvh_probs, cva_probs, avh_probs)

        return data_loader.get_samples_paths(), ['%d\t%.2f\t%.2f\t%.2f' %
                (final_labels[i], cvh_probs[i], cva_probs[i], avh_probs[i])
                for i in range(len(final_labels))]

    def get_samples_results_header(self):
        """ Returns the header of the results summaries saved for the samples in the report. """
        return ['PredictedLabel', 'CvsHProb', 'CvsAProb', 'AvsHProb']

    def get_samples_elements_results_summaries(self, data_loader, dict_of_vars):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of lists of values (one list per sample in which a value per element).
        """
        raise Exception('Not applicable here.')
        return None
