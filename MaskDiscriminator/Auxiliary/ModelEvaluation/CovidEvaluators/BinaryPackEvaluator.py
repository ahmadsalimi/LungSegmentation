import numpy as np
from Auxiliary.ModelEvaluation.BinaryEvaluator import BinaryEvaluator


class BinaryPackEvaluator(BinaryEvaluator):
    """
    Despite the parent, this class is used for considering packs of different heights
    as independent samples through the whole sample! Max of these packs results
    in the final decision for the whole sample!
    """

    def __init__(self, model, conf):
        """ Model is a predictor, conf is a dictionary containing configurations and settings """
        super(BinaryEvaluator, self).__init__(model, conf)

    def initiate_required_variables(self, data_loader):
        """ Receives as input the data_loader which contains information about the samples,
        Initiates required variables for keeping the ground truth about the samples required for evaluation,
        (e.g. information about data or predictions of the model). """

        vars_dict = dict()

        # sample level labels
        vars_dict['ground_truth'] = np.zeros((data_loader.get_samples_num(),), dtype=int)
        vars_dict['model_preds'] = np.zeros((data_loader.get_samples_num(),), dtype=int)

        return vars_dict

    def update_variables_based_on_the_output_of_the_model(self, data_loader, dict_of_vars, model_output):
        """ Receives as input a data_loader for loading data and a dictionary containing
        the saved variables. Updates the required variables from the outputs of the model."""

        current_batch_sample_indices, _ = data_loader.get_current_batch_indices()

        dict_of_vars['ground_truth'][current_batch_sample_indices] = \
            data_loader.get_sample_labels(current_batch_sample_indices)

        model_preds = model_output[0].cpu().numpy()
        dict_of_vars['model_preds'][current_batch_sample_indices] = \
            np.maximum(
                model_preds,
                dict_of_vars['model_preds'][current_batch_sample_indices])

    def extract_ground_truth_and_predictions_for_batch(self, data_loader, model_output):
        """ Returns the ground truth and the output of the model related to the current batch which
         should be used in evaluation. """

        current_batch_sample_indices, _ = data_loader.get_current_batch_indices()
        return data_loader.get_sample_labels(current_batch_sample_indices), \
               (model_output[0].cpu().numpy() >= 0.5).astype(int)

    def extract_ground_truth_and_predictions(self, data_loader, dict_of_vars):
        """ Reads the ground truth about all of the samples and the predictions of the model
        from the dictionary of variables and returns them"""
        return dict_of_vars['ground_truth'], (dict_of_vars['model_preds'] >= 0.5).astype(int)

    def get_samples_results_summaries(self, data_loader, dict_of_vars):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of strings (One string per sample containing the summaries related
        to the sample.) """

        return ['%.2f' % x for x in dict_of_vars['model_preds']]

    def get_samples_results_header(self):
        """ Returns the header of the results summaries saved for the samples in the report. """
        return ['PositiveClassProbability']

    def get_samples_elements_results_summaries(self, data_loader, dict_of_vars):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of lists of values (one list per sample in which a value per element).
        """
        raise Exception('Not implemented for this class')
    pass
