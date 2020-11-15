import numpy as np
from Auxiliary.ModelEvaluation.Evaluator import Evaluator
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader


class SofteningEvaluator(Evaluator):
    """ Needs left_slices_positive_probability, right_slices_positive_probability,
    left_patches_positive_probability, right_patches_positive_probability in the output dictionary"""

    def __init__(self, model, conf, data_loader):
        super(SofteningEvaluator, self).__init__(model, conf, data_loader)

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

        # sample level labels
        self.ground_truth = np.zeros((data_loader.get_number_of_samples(),), dtype=int)
        self.sample_preds = np.zeros((data_loader.get_number_of_samples(),), dtype=int)

        # Keeping whether slice summaries have been added
        slices_names = self.content_loader_of_interest.get_elements_names()

        # for keeping sample level summaries
        self.samples_summaries = np.zeros((len(slices_names), 5), dtype=int)

    def reset(self):
        # sample level labels
        self.ground_truth = np.zeros((self.data_loader.get_number_of_samples(),), dtype=int)
        self.sample_preds = np.zeros((self.data_loader.get_number_of_samples(),), dtype=int)

        # Keeping whether slice summaries have been added
        slices_names = self.content_loader_of_interest.get_elements_names()

        # for keeping sample level summaries
        self.samples_summaries = np.zeros((len(slices_names), 5), dtype=int)

    def calculate_evaluation_requirements(self, ground_truth, prediction):
        
        pred_label, sample_stats = prediction
        
        n_healthy = np.sum(ground_truth == 0)
        n_diseased = np.sum(ground_truth == 1)
        
        tp = np.sum(np.logical_and(ground_truth == 1, pred_label == 1))
        tn = np.sum(np.logical_and(ground_truth == 0, pred_label == 0))
        fp = np.sum(np.logical_and(ground_truth == 0, pred_label == 1))
        fn = np.sum(np.logical_and(ground_truth == 1, pred_label == 0))

        h_avg_n_d_slices = np.mean(sample_stats[ground_truth == 0, 0])
        d_avg_n_d_slices = np.mean(sample_stats[ground_truth == 1, 0])
        h_avg_n_d_patches = np.mean(sample_stats[ground_truth == 0, 1])
        d_avg_n_d_patches = np.mean(sample_stats[ground_truth == 1, 1])
        h_avg_n_single_d_slices = np.mean(sample_stats[ground_truth == 0, 2])
        d_avg_n_single_d_slices = np.mean(sample_stats[ground_truth == 1, 2])
        h_avg_n_maligned_d_patches = np.mean(sample_stats[ground_truth == 0, 3])
        d_avg_n_maligned_d_patches = np.mean(sample_stats[ground_truth == 1, 3])
        
        return np.asarray([n_healthy, n_diseased, h_avg_n_d_slices, d_avg_n_d_slices,
                           h_avg_n_d_patches, d_avg_n_d_patches, 
                           h_avg_n_single_d_slices, d_avg_n_single_d_slices,
                           h_avg_n_maligned_d_patches, d_avg_n_maligned_d_patches,
                           tp, tn, fp, fn])

    def aggregate_evaluation_requirements(self,
                                          aggregated_evaluation_requirements, new_evaluation_requirements):
        """ Receives as input 2 numpy arrays of length 8 containing [TP, TN, FP, FN]
        for both negative and positive classes,
        one for the aggregated values and another for the new values to aggregate.
        Returns the new aggregated values. """
        
        w_mean = (lambda v1, n1, v2, n2: (n1 * v1 + n2 * v2) / (n1 + n2))
        
        n_h_1 = aggregated_evaluation_requirements[0]
        n_d_1 = aggregated_evaluation_requirements[1]

        n_h_2 = new_evaluation_requirements[0]
        n_d_2 = new_evaluation_requirements[1]
        
        h_reqs = [
            w_mean(
                aggregated_evaluation_requirements[2 * i + 2], n_h_1, 
                new_evaluation_requirements[2 * i + 2], n_h_2) 
            for i in range(4)]

        d_reqs = [
            w_mean(
                aggregated_evaluation_requirements[2 * i + 3], n_d_1,
                new_evaluation_requirements[2 * i + 3], n_d_2)
            for i in range(4)]
        
        reqs = [[h_reqs[i], d_reqs[i]] for i in range(4)]
        reqs = [x for y in reqs for x in y]
        
        return np.asarray([n_h_1 + n_h_2, n_d_1 + n_d_2] + reqs + \
               list(aggregated_evaluation_requirements[-4:] + new_evaluation_requirements[-4:]))

    def get_the_number_of_evaluation_requirements(self):
        """ Returns the number of stats that are needed to be kept to calculate the final
        evaluation metrics. """
        return 14

    def calculate_evaluation_metrics(self, aggregated_evaluation_requirements):

        _, _, h_avg_n_d_slices, d_avg_n_d_slices, \
            h_avg_n_d_patches, d_avg_n_d_patches, \
            h_avg_n_single_d_slices, d_avg_n_single_d_slices, \
            h_avg_n_maligned_d_patches, d_avg_n_maligned_d_patches, \
            tp, tn, fp, fn = tuple(aggregated_evaluation_requirements)

        if self.conf['phase'] != 'train':
            print('TP, TN, FP, FN: ', tp, tn, fp, fn)

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

        return np.asarray([h_avg_n_d_slices, d_avg_n_d_slices,
                           h_avg_n_d_patches, d_avg_n_d_patches,
                           h_avg_n_single_d_slices, d_avg_n_single_d_slices,
                           h_avg_n_maligned_d_patches, d_avg_n_maligned_d_patches] +
                          calculate_binary_metrics(tp, tn, fp, fn))

    def get_headers_of_evaluation_metrics(self):
        """ Returns a list containing the titles of metrics respectively.
        The average number of positives for positive and negative samples for the model
        and the assigned labels by preprocessing per slice respectively +
        accuracy, sensitivity and specificity for each class respectively."""

        return [
            'H_avg_n_D_slices', 'D_avg_n_D_slices',
            'H_avg_n_D_patches', 'D_avg_n_D_patches',
            'H_avg_n_single_D_slices', 'D_avg_n_single_D_slices',
            'H_avg_n_maligned_D_patches', 'D_avg_n_maligned_D_patches',
            'Accuracy', 'Sens', 'Spec'
        ]

    def update_variables_based_on_the_output_of_the_model(self, model_output):
        """ Receives as input a data_loader for loading data and a dictionary containing
        the saved variables. Updates the required variables from the outputs of the model."""

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_slice_indices = self.data_loader.get_current_batch_elements_indices()

        left_slice_inds, right_slice_inds = \
            self.content_loader_of_interest.get_left_and_right_indices_for_batch(
                current_batch_sample_indices, current_batch_slice_indices)

        self.ground_truth[current_batch_sample_indices] = \
            self.data_loader.get_samples_labels()[current_batch_sample_indices]

        left_slices_positive_probability = (model_output['left_slices_positive_probability'].cpu().numpy() >= 0.5).astype(int)
        right_slices_positive_probability = (model_output['right_slices_positive_probability'].cpu().numpy() >= 0.5).astype(int)
        left_patches_positive_probability = (model_output['left_patches_positive_probability'].cpu().numpy() >= 0.5).astype(int)
        right_patches_positive_probability = (model_output['right_patches_positive_probability'].cpu().numpy() >= 0.5).astype(int)

        # updating probability of slices
        for bsi in range(len(current_batch_sample_indices)):
            # adding the not-added labels!

            sp_left_slices = left_slices_positive_probability[bsi][left_slice_inds[bsi] > -1]
            sp_right_slices = right_slices_positive_probability[bsi][right_slice_inds[bsi] > -1]

            if len(sp_left_slices) > 0:
                self.sample_preds[current_batch_sample_indices[bsi]] = \
                    max(self.sample_preds[current_batch_sample_indices[bsi]], np.amax(sp_left_slices))
            if len(sp_right_slices) > 0:
                self.sample_preds[current_batch_sample_indices[bsi]] = \
                    max(self.sample_preds[current_batch_sample_indices[bsi]], np.amax(sp_right_slices))

            sp_left_patches = left_patches_positive_probability[bsi][left_slice_inds[bsi] > -1, :, :]
            sp_right_patches = right_patches_positive_probability[bsi][right_slice_inds[bsi] > -1, :, :]

            # counting misalignments!!!
            if len(sp_left_slices) > 0:
                c_sp_left_slices = np.concatenate(([0], sp_left_slices, [0]), axis=0)
                c_sp_left_slices = (c_sp_left_slices[:-2] + c_sp_left_slices[2:]) >= 1
                single_left_cnt = np.sum(np.logical_and(sp_left_slices, np.logical_not(c_sp_left_slices)))
            else:
                single_left_cnt = 0

            if len(sp_right_slices) > 0:
                c_sp_right_slices = np.concatenate(([0], sp_right_slices, [0]), axis=0)
                c_sp_right_slices = (c_sp_right_slices[:-2] + c_sp_right_slices[2:]) >= 1
                single_right_cnt = np.sum(np.logical_and(sp_right_slices, np.logical_not(c_sp_right_slices)))
            else:
                single_right_cnt = 0

            if len(sp_left_slices) > 0:
                patch_shape = sp_left_patches[0].shape
            else:
                patch_shape = sp_right_patches[0].shape

            if len(sp_left_slices) > 0:
                c_sp_left_patches = np.concatenate((
                    np.zeros((1,) + patch_shape), sp_left_patches, np.zeros((1,) + patch_shape)), axis=0)
                c_sp_left_patches = (c_sp_left_patches[:-2, :, :] + c_sp_left_patches[2:, :, :]) >= 1
                maligned_left_cnt = np.sum(np.logical_and(sp_left_patches, np.logical_not(c_sp_left_patches)))
            else:
                maligned_left_cnt = 0

            if len(sp_right_slices) > 0:
                c_sp_right_patches = np.concatenate((
                    np.zeros((1,) + patch_shape), sp_right_patches, np.zeros((1,) + patch_shape)), axis=0)
                c_sp_right_patches = (c_sp_right_patches[:-2, :, :] + c_sp_right_patches[2:, :, :]) >= 1
                maligned_right_cnt = np.sum(np.logical_and(sp_right_patches, np.logical_not(c_sp_right_patches)))
            else:
                maligned_right_cnt = 0

            self.samples_summaries[current_batch_sample_indices[bsi], :] += np.asarray([
                    np.sum(sp_left_slices) + np.sum(sp_right_slices),
                    np.sum(sp_left_patches) + np.sum(sp_right_patches),
                    single_left_cnt + single_right_cnt,
                    maligned_left_cnt + maligned_right_cnt,
                    np.sum(left_slice_inds[bsi] > -1) + np.sum(right_slice_inds[bsi] > -1)])

    def extract_ground_truth_and_predictions_for_batch(self, model_output):
        """ Returns the ground truth and the output of the model related to the current batch which
         should be used in evaluation. """

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_slice_indices = self.data_loader.get_current_batch_elements_indices()

        left_slice_inds, right_slice_inds = \
            self.content_loader_of_interest.get_left_and_right_indices_for_batch(
                current_batch_sample_indices, current_batch_slice_indices)

        ground_truth = self.data_loader.get_samples_labels()[current_batch_sample_indices]

        left_slices_positive_probability = (
                    model_output['left_slices_positive_probability'].cpu().numpy() >= 0.5).astype(int)
        right_slices_positive_probability = (
                    model_output['right_slices_positive_probability'].cpu().numpy() >= 0.5).astype(int)
        left_patches_positive_probability = (
                    model_output['left_patches_positive_probability'].cpu().numpy() >= 0.5).astype(int)
        right_patches_positive_probability = (
                    model_output['right_patches_positive_probability'].cpu().numpy() >= 0.5).astype(int)

        sample_stats = np.zeros((len(current_batch_sample_indices), 4))
        sample_preds = np.zeros((len(current_batch_sample_indices),))

        # updating probability of slices
        for bsi in range(len(current_batch_sample_indices)):
            # adding the not-added labels!

            sp_left_slices = left_slices_positive_probability[bsi][left_slice_inds[bsi] > -1]
            sp_right_slices = right_slices_positive_probability[bsi][right_slice_inds[bsi] > -1]

            if len(sp_left_slices) > 0:
                sample_preds[bsi] = max(np.amax(sp_left_slices), sample_preds[bsi])
            if len(sp_right_slices) > 0:
                sample_preds[bsi] = max(np.amax(sp_right_slices), sample_preds[bsi])

            sp_left_patches = left_patches_positive_probability[bsi][left_slice_inds[bsi] > -1, :, :]
            sp_right_patches = right_patches_positive_probability[bsi][right_slice_inds[bsi] > -1, :, :]

            # counting misalignments!!!
            if len(sp_left_slices) > 0:
                c_sp_left_slices = np.concatenate(([0], sp_left_slices, [0]), axis=0)
                c_sp_left_slices = (c_sp_left_slices[:-2] + c_sp_left_slices[2:]) >= 1
                single_left_cnt = np.sum(np.logical_and(sp_left_slices, np.logical_not(c_sp_left_slices)))
            else:
                single_left_cnt = 0

            if len(sp_right_slices) > 0:
                c_sp_right_slices = np.concatenate(([0], sp_right_slices, [0]), axis=0)
                c_sp_right_slices = (c_sp_right_slices[:-2] + c_sp_right_slices[2:]) >= 1
                single_right_cnt = np.sum(np.logical_and(sp_right_slices, np.logical_not(c_sp_right_slices)))
            else:
                single_right_cnt = 0

            if len(sp_left_slices) > 0:
                patch_shape = sp_left_patches[0].shape
            else:
                patch_shape = sp_right_patches[0].shape

            if len(sp_left_slices) > 0:
                c_sp_left_patches = np.concatenate((
                    np.zeros((1,) + patch_shape), sp_left_patches, np.zeros((1,) + patch_shape)), axis=0)
                c_sp_left_patches = (c_sp_left_patches[:-2, :, :] + c_sp_left_patches[2:, :, :]) >= 1
                maligned_left_cnt = np.sum(np.logical_and(sp_left_patches, np.logical_not(c_sp_left_patches)))
            else:
                maligned_left_cnt = 0

            if len(sp_right_slices) > 0:
                c_sp_right_patches = np.concatenate((
                    np.zeros((1,) + patch_shape), sp_right_patches, np.zeros((1,) + patch_shape)), axis=0)
                c_sp_right_patches = (c_sp_right_patches[:-2, :, :] + c_sp_right_patches[2:, :, :]) >= 1
                maligned_right_cnt = np.sum(np.logical_and(sp_right_patches, np.logical_not(c_sp_right_patches)))
            else:
                maligned_right_cnt = 0

            n_slices = np.sum(left_slice_inds[bsi] > -1) + np.sum(right_slice_inds[bsi] > -1)

            sample_stats[bsi] += np.asarray([
                1.0 * (np.sum(sp_left_slices) + np.sum(sp_right_slices)) / n_slices,
                1.0 * (np.sum(sp_left_patches) + np.sum(sp_right_patches)) / n_slices,
                1.0 * (single_left_cnt + single_right_cnt) / n_slices,
                1.0 * (maligned_left_cnt + maligned_right_cnt) / n_slices]
                )

        return ground_truth, (sample_preds, sample_stats)

    def get_ground_truth_and_predictions_for_all_samples(self):
        """ Reads the ground truth about all of the samples and the predictions of the model
        from the dictionary of variables and returns them"""

        samples_summaries = 1.0 * self.samples_summaries / (self.samples_summaries[:, -1])[:, np.newaxis]
        samples_summaries[:, -1] = self.samples_summaries[:, -1]

        return self.ground_truth, (self.sample_preds, samples_summaries)

    def get_samples_results_summaries(self):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of strings (One string per sample containing the summaries related
        to the sample.) """

        samples_summaries = 1.0 * self.samples_summaries / (self.samples_summaries[:, -1])[:, np.newaxis]
        samples_summaries[:, -1] = self.samples_summaries[:, -1]

        return self.data_loader.get_samples_names(), [
            '\t'.join(['%d' % x for x in list(samples_summaries[i])])
            for i in range(len(samples_summaries))]

    def get_samples_results_header(self):
        """ Returns the header of the results summaries saved for the samples in the report. """
        return ['Avg_D_slices', 'Avg_D_patches', 'Avg_single_slices', 'Avg_maligned_patches', 'N_slices']

    def get_samples_elements_results_summaries(self):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of lists of values (one list per sample in which a value per element).
        """
        raise Exception('Not applicable here.')
        return None
