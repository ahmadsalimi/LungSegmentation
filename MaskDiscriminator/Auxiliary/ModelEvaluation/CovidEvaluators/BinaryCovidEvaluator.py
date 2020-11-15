import numpy as np
from Auxiliary.ModelEvaluation.BinaryEvaluator import BinaryEvaluator
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
from os import path, makedirs


class BinaryCovidEvaluator(BinaryEvaluator):

    def __init__(self, model, conf, data_loader):
        super(BinaryCovidEvaluator, self).__init__(model, conf, data_loader)

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

        self.sample_elements_positive_class_probs = \
            [np.zeros((len(elements_names),), dtype=float) for elements_names in
             self.content_loader_of_interest.get_elements_names()]

        if 'presaved_slice_probs_dir' in self.conf and self.data_loader.sample_specification == 'train':
            self.data_loader.initiate_model_preds_for_samples_elements(
                self.content_loader_of_interest.load_slice_probs(self.conf['presaved_slice_probs_dir'])
            )
        else:
            self.data_loader.initiate_model_preds_for_samples_elements(
                [np.zeros((len(x),), dtype=float)
                 for x in self.content_loader_of_interest.samples_slices_paths]
            )

    def reset(self):
        self.ground_truth = np.zeros((self.data_loader.get_number_of_samples(),), dtype=int)

        self.sample_elements_positive_class_probs = \
            [np.zeros((len(elements_names),), dtype=float) for elements_names in
             self.content_loader_of_interest.get_elements_names()]

    def update_variables_based_on_the_output_of_the_model(self, model_output):
        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_elements_indices = self.data_loader.get_current_batch_elements_indices()

        self.ground_truth[current_batch_sample_indices] = \
            self.data_loader.get_samples_labels()[current_batch_sample_indices]

        elements_positive_class_probs = \
            model_output['positive_class_probability_for_elements'].cpu().numpy()

        new_preds = np.zeros((len(current_batch_sample_indices),))
        for bsi in range(len(current_batch_sample_indices)):

            # mask of the elements in range
            ok_mask = (current_batch_elements_indices[bsi] >= 0)
            self.sample_elements_positive_class_probs[
                current_batch_sample_indices[bsi]][current_batch_elements_indices[bsi][ok_mask]] = \
                (elements_positive_class_probs[bsi, :])[ok_mask]

            # updating probabilities of the elements in content loader!
            if self.conf['phase'] == 'train':
                self.data_loader.update_model_preds_for_samples_elements(elements_positive_class_probs)

                new_preds[bsi] = np.amax(self.data_loader.model_preds_for_smaples_elements[current_batch_sample_indices[bsi]])

        if self.conf['phase'] == 'train':
            self.data_loader.update_model_preds_for_sample(new_preds)

    def extract_ground_truth_and_predictions_for_batch(self, model_output):
        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_elements_indices = self.data_loader.get_current_batch_elements_indices()

        gt = self.data_loader.get_samples_labels()[current_batch_sample_indices]

        elements_positive_class_probs = \
            model_output['positive_class_probability_for_elements'].cpu().numpy()

        sample_preds = np.asarray([
            np.amax((elements_positive_class_probs[i, :])[current_batch_elements_indices[i, :] > -1])
            for i in range(len(current_batch_sample_indices))])

        return gt, (sample_preds >= 0.5).astype(int)

    def calculate_softened_probabilities(self):
        """ Returns 2 arrays, elements names and their softened probabilities
        based on neighboring elements (for covid for now!), separately on each lobe. """

        def soften_probs(snames, sprobs):
            left_mask = np.asarray(['left' in sn for sn in snames])
            left_probs = sprobs[left_mask]
            right_probs = sprobs[np.logical_not(left_mask)]

            left_slices_names = np.asarray([sn for sn in snames if 'left' in sn])
            right_slices_names = np.asarray([sn for sn in snames if 'right' in sn])

            # softening
            def soften(arr, ind):
                neighbs = []
                if ind > 0:
                    neighbs.append(arr[ind - 1])
                if ind < len(arr) - 1:
                    neighbs.append(arr[ind + 1])

                if len(neighbs) == 0:
                    return arr[ind]

                return min(max(neighbs), arr[ind])

            soft_left = np.vectorize(lambda ind: soften(left_probs, ind))
            soft_right = np.vectorize(lambda ind: soften(right_probs, ind))

            if len(left_probs) > 0:
                softened_left_probs = soft_left(np.arange(len(left_probs)))
            else:
                softened_left_probs = np.asarray([])

            if len(right_probs) > 0:
                softened_right_probs = soft_right(np.arange(len(right_probs)))
            else:
                softened_right_probs = np.asarray([])

            return [np.concatenate((left_slices_names, right_slices_names), axis=0),
                    np.concatenate((softened_left_probs, softened_right_probs), axis=0)]

        samples_slices_probs = self.sample_elements_positive_class_probs
        slices_paths = self.content_loader_of_interest.get_elements_names()
        softened_slices_info = []

        # for each sample
        for i in range(len(slices_paths)):
            softened_slices_info.append(soften_probs(slices_paths[i], samples_slices_probs[i]))

        return softened_slices_info

    def get_ground_truth_and_predictions_for_all_samples(self):
        softened_probs_info = self.calculate_softened_probabilities()
        softened_model_preds = np.asarray([np.amax(softened_probs_info[i][1]) for i in range(len(softened_probs_info))])
        return self.ground_truth, (softened_model_preds >= 0.5).astype(int)

    def get_samples_results_summaries(self):

        samples_names = self.data_loader.get_samples_names()

        softened_slices_probs_info = self.calculate_softened_probabilities()
        softened_model_probs = [np.amax(x[1]) for x in softened_slices_probs_info]

        return samples_names, ['%.2f' % x for x in softened_model_probs]

    def get_samples_results_header(self):
        return ['PositiveClassProbability']

    def get_samples_elements_results_summaries(self):
        softened_slices_probs_info = self.calculate_softened_probabilities()

        return [[(softened_slices_probs_info[i][0][j], softened_slices_probs_info[i][1][j])
                 for j in range(len(softened_slices_probs_info[i][0]))]
                for i in range(len(softened_slices_probs_info))]

    def save_output_of_the_model_for_batch(self, model_output):

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_slices_indices = self.data_loader.get_current_batch_elements_indices()
        sample_slices_paths = self.content_loader_of_interest.samples_slices_paths
        sample_paths = self.content_loader_of_interest.get_samples_names()

        slices_org_imgs = self.content_loader_of_interest.prepare_batch_preprocessed_untouched_images(self.data_loader.batch_chooser)

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
