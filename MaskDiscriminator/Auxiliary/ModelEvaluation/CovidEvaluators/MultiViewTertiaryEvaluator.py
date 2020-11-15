import numpy as np
from Auxiliary.ModelEvaluation.CovidEvaluators.TertiaryCovidEvaluator import TertiaryCovidEvaluator


class MultiViewTertiaryEvaluator(TertiaryCovidEvaluator):

    def __init__(self, model, conf, data_loader):
        """ Model is a predictor, conf is a dictionary containing configurations and settings """
        super(MultiViewTertiaryEvaluator, self).__init__(model, conf, data_loader)

    def get_ground_truth_and_predictions_for_all_samples(self):
        """ Reads the ground truth about all of the samples and calculates the probabilities for
        corona and abnormality based on the predictions of the model and returns them.
        Returns an array for ground truth and a tuple of two arrays for corona and abnormal probabilities."""

        views_corona_probs, views_abn_probs = self.extract_disease_probs()

        # Aggregating results of different views
        views_base_paths, views_indices = self.content_loader_of_interest.get_views_indices()

        corona_prob = np.zeros((len(views_base_paths),), dtype=float)
        abn_prob = np.zeros((len(views_base_paths),), dtype=float)
        ground_truths = np.zeros((len(views_base_paths),), dtype=int)

        for i in range(len(views_base_paths)):

            corona_prob[i] = np.amax(views_corona_probs[views_indices[i]])
            abn_prob[i] = np.amax(views_abn_probs[views_indices[i]])
            ground_truths[i] = np.amax(self.ground_truth[views_indices[i]])

        return ground_truths, ((corona_prob >= 0.5).astype(int), (abn_prob >= 0.5).astype(int))

    def get_samples_results_summaries(self):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of strings (One string per sample containing the summaries related
        to the sample.) """

        views_corona_probs, views_abn_probs = \
            self.extract_disease_probs()

        views_paths = self.content_loader_of_interest.get_samples_names()
        views_masks_n = self.content_loader_of_interest.get_all_masks_num()

        # Aggregating results of different views
        views_base_paths, views_indices = self.content_loader_of_interest.get_views_indices()

        def get_sample_summary_str(si):
            corona_prob = np.amax(views_corona_probs[views_indices[si]])
            abn_prob = np.amax(views_abn_probs[views_indices[si]])
            n_views = len(views_indices[si])
            n_views_with_corona_prob = np.sum((views_corona_probs[views_indices[si]] >= 0.5).astype(int))
            n_views_with_abn_prob = np.sum((views_abn_probs[views_indices[si]] >= 0.5).astype(int))
            views_reports = ','.join([
                '%s:%d:%.2f:%.2f' % (views_paths[vi], views_masks_n[vi], views_corona_probs[vi], views_abn_probs[vi])
                for vi in views_indices[si]])

            return '%.2f\t%.2f\t%d\t%d\t%d\t%s' % \
                   (corona_prob, abn_prob, n_views,
                    n_views_with_corona_prob, n_views_with_abn_prob, views_reports)

        return views_base_paths, [get_sample_summary_str(i) for i in range(len(views_base_paths))]

    def get_samples_results_header(self):
        """ Returns the header of the results summaries saved for the samples in the report. """
        return ['CoronaProb', 'AbnProb', 'N_views',
                'N_ViewsWithCoronaProbability', 'N_ViewsWithAbnProbability',
                'ViewsProbabilities']
