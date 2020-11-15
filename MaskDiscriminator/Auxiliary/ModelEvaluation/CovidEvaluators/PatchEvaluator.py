import numpy as np
from Auxiliary.ModelEvaluation.Evaluator import Evaluator
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from os import path, makedirs


class PatchEvaluator(Evaluator):
    """ Needs positive_class_probability_for_patches in the output dictionary"""

    def __init__(self, model, conf, data_loader):
        super(PatchEvaluator, self).__init__(model, conf, data_loader)

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

        # sample level labels
        self.ground_truth = np.zeros((data_loader.get_number_of_samples(),), dtype=int)

        # Keeping whether slice summaries have been added
        slices_names = self.content_loader_of_interest.get_elements_names()
        self.has_added_slice_summaries = [np.full((len(ss),), False) for ss in slices_names]

        # for keeping sample level summaries
        self.samples_summaries = np.zeros((len(slices_names), 5), dtype=int)

    def reset(self):
        # sample level labels
        self.ground_truth = np.zeros((self.data_loader.get_number_of_samples(),), dtype=int)

        # Keeping whether slice summaries have been added
        slices_names = self.content_loader_of_interest.get_elements_names()
        self.has_added_slice_summaries = [np.full((len(ss),), False) for ss in slices_names]

        # for keeping sample level summaries
        self.samples_summaries = np.zeros((len(slices_names), 5), dtype=int)

    def calculate_evaluation_requirements(self, ground_truth, prediction):
        """ Receives as input ground truth which is a tuple of two numpy arrays
        (sample_labels and one array with the same shape as prediction) and
        prediction which is a numpy array.
        For simplicity, 2 types of predictions are accepted:
        1. Batch form: Batch x Slice x width x height
        2. Summarized form: Sample x 5 (n_slice, tp, tn, fp, fn for the sample)
        Calculates and returns the number of
        [NSlices, NTP, NTN, NFP, NFN, PSlices, PTP, PTN, PFP, PFN] in one numpy array
        which are prediction stats for the patches of the negative samples and the positive samples."""

        sample_label, patch_labels = ground_truth

        stats = []
        for c in [0, 1]:
            if len(list(prediction.shape)) == 2:
                stats += list(np.sum(prediction[sample_label == c], axis=0))
            else:
                tp = np.sum(
                    (sample_label == c).astype(int)[:, np.newaxis, np.newaxis, np.newaxis] *
                    np.logical_and(patch_labels == 1, prediction == 1)
                )
                tn = np.sum(
                    (sample_label == c).astype(int)[:, np.newaxis, np.newaxis, np.newaxis] *
                    np.logical_and(patch_labels == 0, prediction == 0)
                )
                fp = np.sum(
                    (sample_label == c).astype(int)[:, np.newaxis, np.newaxis, np.newaxis] *
                    np.logical_and(patch_labels == 0, prediction == 1)
                )
                fn = np.sum(
                    (sample_label == c).astype(int)[:, np.newaxis, np.newaxis, np.newaxis] *
                    np.logical_and(patch_labels == 1, prediction == 0)
                )

                n_slices = np.sum(sample_label == c) * prediction.shape[1]
                stats += [n_slices, tp, tn, fp, fn]

        return np.asarray(stats)

    def aggregate_evaluation_requirements(self,
                                          aggregated_evaluation_requirements, new_evaluation_requirements):
        """ Receives as input 2 numpy arrays of length 8 containing [TP, TN, FP, FN]
        for both negative and positive classes,
        one for the aggregated values and another for the new values to aggregate.
        Returns the new aggregated values. """
        return aggregated_evaluation_requirements + new_evaluation_requirements

    def get_the_number_of_evaluation_requirements(self):
        """ Returns the number of stats that are needed to be kept to calculate the final
        evaluation metrics. """
        return 10

    def calculate_evaluation_metrics(self, aggregated_evaluation_requirements):
        """ Receives as input a numpy array of length 10 containing [N_Slices, TP, TN, FP, FN]
        for negative and positive classes for the whole data,
        returns an array containing the average number of positives for positive and negative samples for the model
        and the assigned labels by preprocessing per slice respectively +
        accuracy, sensitivity and specificity for each class respectively"""

        ns, ntp, ntn, nfp, nfn, ps, ptp, ptn, pfp, pfn = tuple(aggregated_evaluation_requirements)

        if self.conf['phase'] != 'train':
            print('### NS, NTP, NTN, NFP, NFN, PS, PTP, PTN, PFP, PFN\n', aggregated_evaluation_requirements)

        def calculate_binary_metrics(tp, tn, fp, fn):
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

            return [accuracy, sensitivity, specificity]

        if ps > 0:
            avg_mpp = float(ptp + pfp) / ps
            avg_app = float(ptp + pfn) / ps
        else:
            avg_mpp = -1
            avg_app = -1

        if ns > 0:
            avg_mnp = float(ntp + nfp) / ns
            avg_anp = float(ntp + nfn) / ns
        else:
            avg_mnp = -1
            avg_anp = -1

        return np.asarray([avg_mpp, avg_mnp, avg_app, avg_anp] + \
                          calculate_binary_metrics(ptp, ptn, pfp, pfn) + \
                          calculate_binary_metrics(ntp, ntn, nfp, nfn))

    def get_headers_of_evaluation_metrics(self):
        """ Returns a list containing the titles of metrics respectively.
        The average number of positives for positive and negative samples for the model
        and the assigned labels by preprocessing per slice respectively +
        accuracy, sensitivity and specificity for each class respectively."""
        return ['avg_MPP', 'avg_MNP', 'avg_APP', 'avg_ANP',
                'PAcc', 'PSens', 'PSpec', 'NAcc', 'NSens', 'NSpec']

    def update_variables_based_on_the_output_of_the_model(self, model_output):
        """ Receives as input a data_loader for loading data and a dictionary containing
        the saved variables. Updates the required variables from the outputs of the model."""

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_slice_indices = self.data_loader.get_current_batch_elements_indices()
        current_batch_patch_labels = \
            self.content_loader_of_interest.prepare_batch_patch_infection_labels(
                self.data_loader.batch_chooser)

        self.ground_truth[current_batch_sample_indices] = \
            self.data_loader.get_samples_labels()[current_batch_sample_indices]

        patch_probs = model_output['positive_class_probability_for_patches'].cpu().numpy()
        patch_labels = (patch_probs >= 0.5).astype(int)

        # updating probability of slices
        for bsi in range(len(current_batch_sample_indices)):
            # adding the not-added labels!
            ok_mask = (current_batch_slice_indices[bsi] >= 0)

            slices_patch_summaries = np.stack(tuple([
                np.sum(np.logical_and(
                    current_batch_patch_labels[bsi][ok_mask] == gti,
                    patch_labels[bsi][ok_mask] == pi
                ), axis=(1, 2))
                for (gti, pi) in [(1, 1), (0, 0), (0, 1), (1, 0)]]), axis=1)  # S x 4

            # Appending a one column to be able to count the new slices
            slices_patch_summaries = np.concatenate((
                np.ones((len(slices_patch_summaries), 1), dtype=int),
                slices_patch_summaries), axis=1)

            sample_unadded_summaries = np.sum(
                (1 -
                 self.has_added_slice_summaries
                 [current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi][ok_mask]]
                 ).astype(int)[:, np.newaxis] * slices_patch_summaries
            , axis=0)

            self.samples_summaries[current_batch_sample_indices[bsi]] += sample_unadded_summaries
            self.has_added_slice_summaries[current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi][ok_mask]] = True

    def extract_ground_truth_and_predictions_for_batch(self, model_output):
        """ Returns the ground truth and the output of the model related to the current batch which
         should be used in evaluation. """

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        patch_probs = model_output['positive_class_probability_for_patches'].cpu().numpy()
        patch_labels = (patch_probs >= 0.5).astype(int)

        return \
            (self.data_loader.get_samples_labels()[current_batch_sample_indices],
             self.content_loader_of_interest.prepare_batch_patch_infection_labels(
                 self.data_loader.batch_chooser)
             ), patch_labels

    def get_ground_truth_and_predictions_for_all_samples(self):
        """ Reads the ground truth about all of the samples and the predictions of the model
        from the dictionary of variables and returns them"""

        return (self.ground_truth, self.ground_truth), \
               self.samples_summaries

    def get_samples_results_summaries(self):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of strings (One string per sample containing the summaries related
        to the sample.) """

        samples_summaries = self.samples_summaries

        return self.data_loader.get_samples_names(), [
            '\t'.join(['%d' % x for x in list(samples_summaries[i])]) +
            '\t%.2f\t%.2f' % (
                float(samples_summaries[i, 1] + samples_summaries[i, 3]) / max(1, samples_summaries[i, 0]),
                float(samples_summaries[i, 1] + samples_summaries[i, 4]) / max(1, samples_summaries[i, 0]))
            for i in range(len(samples_summaries))]

    def get_samples_results_header(self):
        """ Returns the header of the results summaries saved for the samples in the report. """
        return ['N_Slices', 'TP', 'TN', 'FP', 'FN', 'Avg_MP', 'Avg_AP']

    def get_samples_elements_results_summaries(self):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of lists of values (one list per sample in which a value per element).
        """
        raise Exception('Not applicable here.')
        return None

    def save_output_of_the_model_for_batch(self, model_output):

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_slices_indices = self.data_loader.get_current_batch_elements_indices()
        sample_slices_paths = self.content_loader_of_interest.samples_slices_paths
        sample_paths = self.content_loader_of_interest.get_samples_names()

        slices_org_imgs = self.content_loader_of_interest.prepare_batch_preprocessed_untouched_images(
            self.data_loader.batch_chooser)

        slices_probs = model_output['positive_class_probability_for_elements'].cpu().numpy()
        patches_infection_probs = model_output['positive_class_probability_for_patches'].cpu().numpy()
        attention_probs = model_output['attention_probs'].cpu().numpy()

        def save_lobe_info(lobe_path, lobe_img, lobe_attention, lobe_inf, lobe_prob):
            sd = lobe_path.replace('..', self.conf['report_dir'])
            np.save(sd + '_patches_attentions.npy', lobe_attention)
            np.save(sd + '_patches_infections.npy', lobe_inf)
            np.save(sd + '_slice_prob.npy', lobe_prob)
            np.save(sd + '_org_img.npy', lobe_img)

        saving_jobs = []

        for bsi in range(len(current_batch_sample_indices)):
            # adding the not-added labels!

            sd = sample_paths[current_batch_sample_indices[bsi]].replace('..', self.conf['report_dir'])
            if not path.exists(sd):
                makedirs(sd)

            if np.sum(current_batch_slices_indices[bsi] > -1):
                saving_jobs += [
                    (sample_slices_paths[current_batch_sample_indices[bsi]][current_batch_slices_indices[bsi][bssi]],
                     slices_org_imgs[bsi, bssi, :, :], attention_probs[bsi, bssi, :, :],
                     patches_infection_probs[bsi, bssi, :, :], slices_probs[bsi, bssi])
                    for bssi in range(len(current_batch_slices_indices[bsi]))
                    if current_batch_slices_indices[bsi][bssi] > -1]

        self.savers.run_func(save_lobe_info, saving_jobs)
