import numpy as np
from Auxiliary.ModelEvaluation.BinaryEvaluator import BinaryEvaluator
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader


class MultiViewBinaryEvaluator(BinaryEvaluator):

    def __init__(self, model, conf, data_loader):
        """ Model is a predictor, conf is a dictionary containing configurations and settings """
        super(MultiViewBinaryEvaluator, self).__init__(model, conf, data_loader)

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

    def get_ground_truth_and_predictions_for_all_samples(self):
        """ Reads the ground truth about all of the samples and the predictions of the model
        from the dictionary of variables and returns them"""

        # Aggregating results of different views
        views_base_paths, views_indices = self.content_loader_of_interest.get_views_indices()

        positive_class_prob = self.model_positive_class_probs
        ground_truths = self.ground_truth

        for i in range(len(views_base_paths)):
            positive_class_prob[i] = np.amax(positive_class_prob[views_indices[i]])
            ground_truths[i] = np.amax(self.ground_truth[views_indices[i]])

        return ground_truths, (positive_class_prob >= 0.5).astype(int)

    def get_samples_results_summaries(self):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of strings (One string per sample containing the summaries related
        to the sample. Fields are tab delimited in the string.) """

        positive_class_prob = self.model_positive_class_probs

        views_paths = self.content_loader_of_interest.get_samples_paths()
        views_masks_n = self.content_loader_of_interest.get_all_masks_num()

        # Aggregating results of different views
        views_base_paths, views_indices = self.content_loader_of_interest.get_views_indices()

        def get_sample_summary_str(si):
            positive_prob = np.amax(positive_class_prob[views_indices[si]])
            n_views = len(views_indices[si])
            views_positive_probs = positive_class_prob[views_indices[si]]
            n_views_with_positive_prob = np.sum((views_positive_probs >= 0.5).astype(int))
            views_reports = ','.join([
                '%s:%d:%.2f' % (views_paths[vi], views_masks_n[vi], positive_class_prob[vi])
                for vi in views_indices[si]])

            return '%.2f\t%d\t%d\t%s' % (positive_prob, n_views, n_views_with_positive_prob, views_reports)

        return views_base_paths, [get_sample_summary_str(i) for i in range(len(views_base_paths))]

    def get_samples_results_header(self):
        """ Returns the header of the results summaries saved for the samples in the report. """
        return ['PositiveClassProbability', 'N_views', 'N_ViewsWithPositiveProbability', 'ViewsPositiveProbabilities']
