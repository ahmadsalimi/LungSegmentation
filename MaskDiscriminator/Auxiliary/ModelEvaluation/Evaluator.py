from abc import abstractmethod
import torch
from os import path, makedirs
import numpy as np
from Auxiliary.Threading.WorkerCoordinating import WorkersCoordinator


class Evaluator:

    def __init__(self, model, conf, data_loader):
        """ Model is a predictor, conf is a dictionary containing configurations and settings,
         data_loader is a subclass of type DataLoading.DataLoader which contains information
         about the samples and how to iterate over them. """
        self.model = model
        self.conf = conf
        self.data_loader = data_loader
        
        # A runner for running the model
        self.model_runner = conf['model_runner'](model, conf)

        self.savers = WorkersCoordinator(4)

    @abstractmethod
    def calculate_evaluation_requirements(self, ground_truth, prediction):
        """ Receives as input 2 data structures, one containing information about the ground truth
         and the other about the predictions of the model which are required for calculating
        summarized values which can be used to calculate evaluation metrics.
        Returns the mentioned summarized values.
        This function has been separated from calculating evaluation metrics
        to let the model being run on data in several iterations, so this function
        can be used in each iteration to summarize the requirements
        and the values can be aggregated after the loop."""

    @abstractmethod
    def aggregate_evaluation_requirements(self,
                                          aggregated_evaluation_requirements, new_evaluation_requirements):
        """ Receives as input 2 data structures containing information about the aggregated metrics
        required for the evaluation and the new metrics calculated for one subset of data.
        Returns the new aggregated values. """

    @abstractmethod
    def get_the_number_of_evaluation_requirements(self):
        """ Returns the number of stats that are needed to be kept to calculate the final
        evaluation metrics. """

    @abstractmethod
    def calculate_evaluation_metrics(self, aggregated_evaluation_requirements):
        """ Receives as input the aggregated metrics for the whole data,
        calculates the evaluation metrics and return them in one array"""

    @abstractmethod
    def get_headers_of_evaluation_metrics(self):
        """ Returns a list containing the titles of metrics respectively"""

    @abstractmethod
    def reset(self):
        """ Resets the held information for a new evaluation round!"""

    @abstractmethod
    def update_variables_based_on_the_output_of_the_model(self, model_output):
        """ Receives the output of the model which is a dictionary.
        Extracts and saves the required information from the output of the model."""

    @abstractmethod
    def save_output_of_the_model_for_batch(self, model_output):
        """ Saves the required information from the output of the model, used in
        model output saving mode."""

    @abstractmethod
    def extract_ground_truth_and_predictions_for_batch(self, model_output):
        """ Receives the output of the model which is a dictionary.
        Extracts information required for predicting the label of each sample of the batch,
        returns 2 numpy arrays of ground truths and model's predictions for the batch.
        Current batch is accessible via data_loader field."""

    @abstractmethod
    def get_ground_truth_and_predictions_for_all_samples(self):
        """ Returns 1 numpy arrays, one for the label of all of the samples of the data_loader
        and the other for the predicted labels based on the information gathered during evaluations."""

    @abstractmethod
    def get_samples_results_summaries(self):
        """ Returns the list of samples paths and a list for samples summaries in a list of strings (One string per
        sample containing the summaries related to the sample (fields are tab delimited in the string).) """

    @abstractmethod
    def get_samples_results_header(self):
        """ Returns the header of the results summaries saved for the samples in the report. """

    @abstractmethod
    def get_samples_elements_results_summaries(self):
        """ Calculates summary for each sample, returns the
        summaries in a list of lists of tuples (path, values)
         (one list per sample in which a tuple of path and value per element,
         path is where the results would be saved).
        """

    def summarize_for_evaluation_in_each_iteration(self):
        """ Returns true if information required to calculate evaluation
        metrics on data must be summarized in each iteration. In this way no information is saved!
        For general models it would be in training phase for when maximum number of iterations
         is defined (val loader having no end.) But can be different in different evaluators. """
        return self.conf['phase'] == 'train' and self.conf['val_iters_per_epoch'] is not None

    def evaluate(self, title='Evaluation Metrics'):
        """ CAUTION: Resets the data_loader,
        Iterates over the samples (as much as and how data_loader specifies),
        calculates the overall evaluation requirements and prints them.
        Title specified the title of the string for printing evaluation metrics."""

        with torch.no_grad():

            # Running the model in evaluation mode
            if self.conf['turn_on_val_mode']:
                self.model.eval()
                if self.conf['phase'] != 'train':
                    print('On the eval mode')

            # checking if data should be summarized or saved for final evaluation
            summarize_each_data_part = self.summarize_for_evaluation_in_each_iteration()

            # initiating variables for running evaluations
            self.reset()
            self.data_loader.reset()

            if summarize_each_data_part:
                # Having something to aggregate the eval reqs of each itearation!
                aggregated_eval_reqs = np.zeros((self.get_the_number_of_evaluation_requirements(),))

            iters = 0  # keeping the number of iterations

            total_loss = 0.0

            placeholders = self.model_runner.create_placeholders_for_model_input()

            while True:

                self.data_loader.prepare_next_batch()

                # check if the iteration is done
                if self.data_loader.finished_iteration() or \
                    (self.conf['phase'] == 'train' and summarize_each_data_part and
                     iters >= self.conf['val_iters_per_epoch']):
                    break

                # Filling the placeholders for running the model
                self.data_loader.fill_placeholders(placeholders)

                # Running the model
                model_output = self.model_runner.run_model(placeholders)

                # adding loss if the phase is train
                if self.conf['phase'] == 'train':
                    total_loss += float(model_output['loss'])

                # Updating the saved variables
                self.update_variables_based_on_the_output_of_the_model(
                    model_output)

                # Calculating evaluation requirements
                if summarize_each_data_part:
                    aggregated_eval_reqs = self.aggregate_evaluation_requirements(
                        aggregated_eval_reqs,
                        self.calculate_evaluation_requirements(
                            * self.extract_ground_truth_and_predictions_for_batch(model_output)))

                iters += 1
                del model_output

            # Releasing tensor memories
            del placeholders

            total_loss /= iters

            if not summarize_each_data_part:
                ground_truth, model_preds = self.get_ground_truth_and_predictions_for_all_samples()
                aggregated_eval_reqs = self.calculate_evaluation_requirements(ground_truth, model_preds)

            eval_metrics = self.calculate_evaluation_metrics(aggregated_eval_reqs)

            if title is not None:
                if total_loss != 0:
                    print('%s: loss: %.4f, %s' % (title, total_loss, ', '.join([
                        '%s: %.2f' % (eval_title, eval_value) for (eval_title, eval_value) in
                        zip(self.get_headers_of_evaluation_metrics(), eval_metrics)])))
                else:
                    print('%s: %s' % (title, ', '.join([
                        '%s: %.2f' % (eval_title, eval_value) for (eval_title, eval_value) in
                        zip(self.get_headers_of_evaluation_metrics(), eval_metrics)])))

        return eval_metrics, total_loss

    def save_model_outputs(self):
        """ CAUTION: Resets the data_loader,
        Iterates over the samples (as much as and how data_loader specifies),
        runs the model over the samples and saves the results of the output of the model!
         keeps nothing!"""

        with torch.no_grad():

            # Running the model in evaluation mode
            if self.conf['turn_on_val_mode']:
                self.model.eval()

            # initiating variables for running evaluations
            self.reset()
            self.data_loader.reset()

            placeholders = self.model_runner.create_placeholders_for_model_input()

            while True:

                self.data_loader.prepare_next_batch()

                # check if the iteration is done
                if self.data_loader.finished_iteration():
                    break

                # Filling the placeholders for running the model
                self.data_loader.fill_placeholders(placeholders)

                # Running the model
                model_output = self.model_runner.run_model(placeholders)

                # Save the required output
                self.save_output_of_the_model_for_batch(model_output)

                del model_output

            # Releasing tensor memories
            del placeholders

    def create_sample_report(self, report_save_dir):
        """ CAUTION: Resets the data_loader,
        Iterates over the samples, calculates the outputs of the model for each of them
        and saves the summary of the results in a file."""

        with torch.no_grad():

            # Running the model in evaluation mode
            if self.conf['turn_on_val_mode']:
                self.model.eval()
                print('On the eval mode')

            # initiating variables for running evaluations
            self.reset()
            self.data_loader.reset()

            placeholders = self.model_runner.create_placeholders_for_model_input()

            while True:

                self.data_loader.prepare_next_batch()

                # check if the iteration is done
                if self.data_loader.finished_iteration():
                    break

                # Filling the placeholders for running the model
                self.data_loader.fill_placeholders(placeholders)

                # Running the model
                model_output = self.model_runner.run_model(placeholders)

                # Updating the saved variables
                self.update_variables_based_on_the_output_of_the_model(
                    model_output)

                del model_output

            del placeholders
            samples_dirs, samples_eval_summaries = self.get_samples_results_summaries()

            if not path.exists(path.dirname(report_save_dir)):
                makedirs(path.dirname(report_save_dir))

            f = open(report_save_dir, 'w')
            f.write('SamplePath\t' + '\t'.join(self.get_samples_results_header()) + '\n')
            for sdir, seval in zip(samples_dirs, samples_eval_summaries):
                f.write('%s\t%s\n' % (sdir, seval))
            f.close()

    def create_report_for_sample_elements(self, save_dir):
        """ CAUTION: Resets the data_loader,
        Receives as input a data_loader, which is an instance of Datadata_loader containing the samples
        and methods for loading their information and an iterator to iterate over them,
        iterates over the samples, calculates the outputs of the model for each element of each sample
        and saves the results in the given directory. (e.g. used for saving the probabilities of being covid
        assigned to the slices of samples.)"""
        with torch.no_grad():

            # Running the model in evaluation mode
            if self.conf['turn_on_val_mode']:
                self.model.eval()
                print('On the eval mode')

            # initiating variables for running evaluations
            self.reset()
            self.data_loader.reset()
            placeholders = self.model_runner.create_placeholders_for_model_input()

            while True:

                self.data_loader.prepare_next_batch()

                # check if the iteration is done
                if self.data_loader.finished_iteration():
                    break

                # Filling the placeholders for running the model
                self.data_loader.fill_placeholders(placeholders)

                # Running the model
                model_output = self.model_runner.run_model(placeholders)

                # Updating the saved variables
                self.update_variables_based_on_the_output_of_the_model(
                    model_output)

                del model_output

            del placeholders
            samples_elements_eval_summaries = self.get_samples_elements_results_summaries()

            if not path.exists(save_dir):
                makedirs(save_dir)

            for i in range(len(samples_elements_eval_summaries)):
                for sel_path, sel_evals in samples_elements_eval_summaries[i]:
                    new_path = sel_path.replace('..', self.conf['report_dir'])
                    if not path.exists(path.dirname(new_path)):
                        makedirs(path.dirname(new_path))
                    np.save(new_path, sel_evals)

    def create_report_for_sample_and_its_elements(self, save_dir):
        """ CAUTION: Resets the data_loader,
        Receives as input a data_loader, which is an instance of Datadata_loader containing the samples
        and methods for loading their information and an iterator to iterate over them,
        iterates over the samples, calculates the outputs of the model for each element of each sample
        and saves the results in the given directory (a summary for the samples and val information for slices).
        (e.g. used for saving the probabilities of being covid assigned to the slices of samples.)"""
        with torch.no_grad():

            # Running the model in evaluation mode
            if self.conf['turn_on_val_mode']:
                self.model.eval()
                print('On the eval mode')

            # initiating variables for running evaluations
            self.reset()
            self.data_loader.reset()
            placeholders = self.model_runner.create_placeholders_for_model_input()

            while True:

                self.data_loader.prepare_next_batch()

                # check if the iteration is done
                if self.data_loader.finished_iteration():
                    break

                # Filling the placeholders for running the model
                self.data_loader.fill_placeholders(placeholders)

                # Running the model
                model_output = self.model_runner.run_model(placeholders)

                # Updating the saved variables
                self.update_variables_based_on_the_output_of_the_model(
                    model_output)

                del model_output

            del placeholders

            if not path.exists(save_dir):
                makedirs(save_dir)

            # creating report for samples:
            samples_dirs, samples_eval_summaries = self.get_samples_results_summaries()

            sample_report_save_dir = save_dir + '/SamplesSummaries.tsv'

            f = open(sample_report_save_dir, 'w')
            f.write('SamplePath\t' + '\t'.join(self.get_samples_results_header()) + '\n')
            for sdir, seval in zip(samples_dirs, samples_eval_summaries):
                f.write('%s\t%s\n' % (sdir, seval))
            f.close()

            # creating report for sample elements:
            samples_elements_eval_summaries = self.get_samples_elements_results_summaries()

            for i in range(len(samples_elements_eval_summaries)):
                for sel_path, sel_evals in samples_elements_eval_summaries[i]:
                    new_path = save_dir + '/' + '/'.join(sel_path.split('/')[2:])
                    if not path.exists(path.dirname(new_path)):
                        makedirs(path.dirname(new_path))
                    np.save(new_path, sel_evals)
