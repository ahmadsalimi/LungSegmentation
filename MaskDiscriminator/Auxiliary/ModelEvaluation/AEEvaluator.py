import numpy as np
from Auxiliary.ModelEvaluation.Evaluator import Evaluator


class AEEvaluator(Evaluator):

    def __init__(self, model, conf, data_loader):
        super(AEEvaluator, self).__init__(model, conf, data_loader)
        self.sample_sum_loss = np.zeros((data_loader.get_number_of_samples(),), dtype=float)
        self.sample_n_repeats = np.zeros((data_loader.get_number_of_samples(),), dtype=float)

    def calculate_evaluation_requirements(self, ground_truth, prediction):
        return np.asarray([
            len(prediction),
            np.mean(prediction),
            np.mean(prediction * prediction),
            np.amax(prediction)
        ])

    def aggregate_evaluation_requirements(self,
                                          aggregated_evaluation_requirements, new_evaluation_requirements):

        n_old = aggregated_evaluation_requirements[0]
        n_new = new_evaluation_requirements[0]

        return np.asarray([
            n_old + n_new,
            (n_old * aggregated_evaluation_requirements[1] +
             n_new * aggregated_evaluation_requirements[1]) / (n_old + n_new),
            (n_old * aggregated_evaluation_requirements[2] +
             n_new * aggregated_evaluation_requirements[2]) / (n_old + n_new),
            max(aggregated_evaluation_requirements[3], new_evaluation_requirements[3])
        ])

    def get_the_number_of_evaluation_requirements(self):
        return 4

    def calculate_evaluation_metrics(self, aggregated_evaluation_requirements):

        return np.asarray([
            aggregated_evaluation_requirements[1],
            np.sqrt(
                aggregated_evaluation_requirements[2] -
                aggregated_evaluation_requirements[1] ** 2),
            aggregated_evaluation_requirements[3]])

    def get_headers_of_evaluation_metrics(self):
        """ Returns a list containing the titles of metrics respectively"""
        return ['MeanSLoss', 'StdSLoss', 'MaxSLoss']

    def reset(self):
        self.sample_avg_loss = np.zeros((self.data_loader.get_number_of_samples(),), dtype=float)
        self.sample_n_repeats = np.zeros((self.data_loader.get_number_of_samples(),), dtype=float)

    def update_variables_based_on_the_output_of_the_model(self, model_output):
        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()

        sample_loss = model_output['sample_loss'].cpu().numpy()

        np.add.at(self.sample_sum_loss, current_batch_sample_indices, sample_loss)
        np.add.at(self.sample_n_repeats, current_batch_sample_indices, 1)

    def extract_ground_truth_and_predictions_for_batch(self, model_output):
        return None, model_output['sample_loss'].cpu().numpy()

    def get_ground_truth_and_predictions_for_all_samples(self):
        return None, self.sample_sum_loss / self.sample_n_repeats

    def get_samples_results_summaries(self):
        samples_names = self.data_loader.get_samples_names()
        return samples_names, ['%d\t%.2f' %
                               (self.sample_n_repeats[i],
                                self.sample_sum_loss[i] / self.sample_n_repeats[i])
                               for i in range(len(samples_names))]

    def get_samples_results_header(self):
        return ['N_Els', 'AverageLoss']

    def get_samples_elements_results_summaries(self):
        raise Exception('Not applicable to this class')
