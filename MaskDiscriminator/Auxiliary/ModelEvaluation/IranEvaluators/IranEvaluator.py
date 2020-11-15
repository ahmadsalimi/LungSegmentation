from Auxiliary.ModelEvaluation.BinaryEvaluator import BinaryEvaluator
from Auxiliary.DataLoading.ContentLoading.CovidCTLoader import CovidCTLoader
import numpy as np
from os import path, makedirs


class IranEvaluator(BinaryEvaluator):
    
    def __init__(self, model, conf, data_loader):
        super(IranEvaluator, self).__init__(model, conf, data_loader)

        self.content_loader_of_interest = None
        for cl in data_loader.content_loaders:
            if isinstance(cl, CovidCTLoader):
                self.content_loader_of_interest = cl
                break

        if self.content_loader_of_interest is None:
            raise Exception('No known elemented content loader found in data loader.')

    def save_output_of_the_model_for_batch(self, model_output):

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        raw_slice_inds = self.data_loader.get_current_batch_elements_indices()
        sample_slices_paths = self.content_loader_of_interest.samples_slices_paths
        sample_paths = self.content_loader_of_interest.get_samples_names()
        
        slices_org_imgs = self.content_loader_of_interest.prepare_batch_preprocessed_untouched_images(self.data_loader.batch_chooser)

        left_slice_inds, right_slice_inds = \
            self.content_loader_of_interest.get_left_and_right_indices_for_batch(
                current_batch_sample_indices, raw_slice_inds)

        left_patches_attention = model_output['left_patches_attention'].cpu().numpy()
        right_patches_attention = model_output['right_patches_attention'].cpu().numpy()
        image_extracted_features_left = model_output['image_extracted_features_left'].cpu().numpy()
        image_extracted_features_right = model_output['image_extracted_features_right'].cpu().numpy()

        def save_lobe_info(lobe_path, lobe_attention, lobe_features, lobe_org_img):
            sd = lobe_path.replace('..', self.conf['report_dir'])
            np.save(sd + '_org_img.npy', lobe_org_img)
            np.save(sd + '_patches_attentions.npy', lobe_attention)
            np.save(sd + '_patches_features.npy', lobe_features)

        saving_jobs = []

        for bsi in range(len(current_batch_sample_indices)):
            # adding the not-added labels!

            sd = sample_paths[current_batch_sample_indices[bsi]].replace('..', self.conf['report_dir'])
            if not path.exists(sd):
                makedirs(sd)
            
            if np.sum(left_slice_inds[bsi] > -1):
                saving_jobs += [
                    (sample_slices_paths[current_batch_sample_indices[bsi]][left_slice_inds[bsi][bssi]],
                     left_patches_attention[bsi, bssi, :, :], image_extracted_features_left[bsi, bssi, :],
                     slices_org_imgs[bsi, bssi, :, :]) 
                    for bssi in range(len(left_slice_inds[bsi])) if left_slice_inds[bsi][bssi] > -1]

            if np.sum(right_slice_inds[bsi] > -1):
                saving_jobs += [
                    (sample_slices_paths[current_batch_sample_indices[bsi]][right_slice_inds[bsi][bssi]],
                     right_patches_attention[bsi, bssi, :, :], image_extracted_features_right[bsi, bssi, :],
                     slices_org_imgs[bsi, bssi, :, :])
                    for bssi in range(len(right_slice_inds[bsi])) if right_slice_inds[bsi][bssi] > -1]

        self.savers.run_func(save_lobe_info, saving_jobs)

