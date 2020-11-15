import numpy as np
from Auxiliary.ModelEvaluation.CovidEvaluators.PatchEvaluator import PatchEvaluator
import multiprocessing
from multiprocessing import Pool
from os import path, makedirs
from Auxiliary.Threading.WorkerCoordinating import WorkersCoordinator


class PatchInfectionSaver(PatchEvaluator):
    """ This model is used for saving the up-resolutioned infections of the model! in the report_dir directory"""

    def __init__(self, model, conf, data_loader):
        """ Model is a predictor, conf is a dictionary containing configurations and settings """
        super(PatchInfectionSaver, self).__init__(model, conf, data_loader)
        multiprocessing.set_start_method('spawn', force=True)
        self.pool = Pool(8)
        self.savers = WorkersCoordinator(4)

    def update_variables_based_on_the_output_of_the_model(self, model_output):
        """ Receives as input a data_loader for loading data and a dictionary containing
        the saved variables. Updates the required variables from the outputs of the model."""

        #super(PatchInfectionSaver, self).update_variables_based_on_the_output_of_the_model(
        #    model_output)

        # Saving stuff here instead of keeping one image per slice which would result in memory blow out!

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_slice_indices = self.data_loader.get_current_batch_elements_indices()

        slices_paths = self.content_loader_of_interest.get_elements_names()
        patch_probs = model_output['positive_class_probability_for_patches'].cpu().numpy()

        # Upresolutioning patch infections

        b, s, w, h = patch_probs.shape
        up_resolution_patch_probs = self.pool.map(
            self.model.up_infection_resolution,
            list(patch_probs.reshape((b * s, w, h))))
        up_resolution_patch_probs = np.stack(tuple(up_resolution_patch_probs), axis=0)
        up_resolution_patch_probs = up_resolution_patch_probs.reshape((b, s, 256, 256))

        # updating probability of slices
        for bbsi in range(len(current_batch_sample_indices)):
            for bbssi in range(len(current_batch_slice_indices[bbsi])):
                save_dir = self.conf['report_dir'] + '/' + '/'.join(
                    slices_paths[current_batch_sample_indices[bbsi]][current_batch_slice_indices[bbsi][bbssi]].split('/')[2:])
                if not path.exists(path.dirname(save_dir)):
                    makedirs(path.dirname(save_dir))

        save_func = (lambda fbsi, fbssi: np.save(
            self.conf['report_dir'] + '/' + '/'.join(
                slices_paths[current_batch_sample_indices[fbsi]][current_batch_slice_indices[fbsi][fbssi]].split('/')[2:])
            , up_resolution_patch_probs[fbsi][fbssi]))

        self.savers.run_func(save_func, [(bsi, bssi)
                                         for bsi in range(len(current_batch_sample_indices))
                                         for bssi in range(len(current_batch_slice_indices[bsi]))
                                         ])
