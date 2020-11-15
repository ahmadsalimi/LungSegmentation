import numpy as np
from Auxiliary.ModelEvaluation.CovidEvaluators.BinaryCovidEvaluator import BinaryCovidEvaluator
from multiprocessing import Pool
from skimage import morphology


class BinarySuperInfoExtractor(BinaryCovidEvaluator):

    def __init__(self, model, conf, data_loader):
        """ Model is a predictor, conf is a dictionary containing configurations and settings """
        super(BinarySuperInfoExtractor, self).__init__(model, conf, data_loader)

        # A pool for calculating and saving information for slices      
        self.pool = Pool(8)

        # slice level labels
        slices_names = self.content_loader_of_interest.get_elements_names()

        self.slices_probs = [np.zeros((len(ss),)) for ss in slices_names]

        # The size of the slice lobe in pixels
        self.slices_lobe_sizes = [np.zeros((len(ss),)) for ss in slices_names]

        # percentage of the total infections inside lobe mask - pixel level
        self.slices_model_inf = [np.zeros((len(ss),), dtype=float) for ss in slices_names]
        self.slices_model_fine_inf = [np.zeros((len(ss),), dtype=float) for ss in slices_names]
        self.slices_assigned_inf = [np.zeros((len(ss),), dtype=float) for ss in slices_names]
        self.slices_union_inf = [np.zeros((len(ss),), dtype=float) for ss in slices_names]
        self.n_consolidated_middle_lobes = np.zeros((len(slices_names),), dtype=int)
        self.sample_slice_thickness = [np.zeros((len(ss),), dtype=float) for ss in slices_names]

    def reset(self):
        super(BinarySuperInfoExtractor, self).reset()

        # slice level labels
        slices_names = self.content_loader_of_interest.get_elements_names()

        self.slices_probs = [np.zeros((len(ss),)) for ss in slices_names]

        # The size of the slice lobe in pixels
        self.slices_lobe_sizes = [np.zeros((len(ss),)) for ss in slices_names]

        # percentage of the total infections inside lobe mask - pixel level
        self.slices_model_inf = [np.zeros((len(ss),), dtype=float) for ss in slices_names]
        self.slices_model_fine_inf = [np.zeros((len(ss),), dtype=float) for ss in slices_names]
        self.slices_assigned_inf = [np.zeros((len(ss),), dtype=float) for ss in slices_names]
        self.slices_union_inf = [np.zeros((len(ss),), dtype=float) for ss in slices_names]
        self.n_consolidated_middle_lobes = np.zeros((len(slices_names),), dtype=int)
        self.sample_slice_thickness = [np.zeros((len(ss),), dtype=float) for ss in slices_names]

    def update_variables_based_on_the_output_of_the_model(self, model_output):
        """ Receives as input a data_loader for loading data and a dictionary containing
        the saved variables. Updates the required variables from the outputs of the model."""

        current_batch_sample_indices = self.data_loader.get_current_batch_sample_indices()
        current_batch_slice_indices = self.data_loader.get_current_batch_elements_indices()

        batch_org_imgs = self.content_loader_of_interest.prepare_batch_preprocessed_untouched_images(self.data_loader.batch_chooser)
        batch_mask_labels = self.content_loader_of_interest.prepare_batch_mask_labels(self.data_loader.batch_chooser)
        batch_assigned_inf_labels = self.content_loader_of_interest.prepare_batch_pixel_level_inf_labels(self.data_loader.batch_chooser)
        batch_lobe_bounding_boxes = self.content_loader_of_interest.prepare_batch_bounding_boxes(self.data_loader.batch_chooser)
        batch_lobe_relative_locs = self.content_loader_of_interest.prepare_batch_slice_relative_location(self.data_loader.batch_chooser)
        batch_lobe_thickness = self.content_loader_of_interest.prepare_batch_slice_thickness(self.data_loader.batch_chooser)

        self.ground_truth[current_batch_sample_indices] = self.data_loader.get_samples_labels()[current_batch_sample_indices]

        slice_probs = model_output['positive_class_probability_for_elements'].cpu().numpy()
        patch_probs = model_output['positive_class_probability_for_patches'].cpu().numpy()

        patch_probs = (patch_probs >= 0.5).astype(int)
        batch_assigned_inf_labels = (batch_assigned_inf_labels >= 0.5).astype(int)

        inf_handler = TopLevelInfHandler(self.conf)

        samples_slices_info = self.pool.map(calculate_slices_info, [(
            inf_handler, current_batch_slice_indices[bsi],
           self.slices_lobe_sizes[current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi]],
            batch_mask_labels[bsi], batch_assigned_inf_labels[bsi], 
            batch_lobe_bounding_boxes[bsi], patch_probs[bsi],
            batch_org_imgs[bsi]
        ) for bsi in range(len(current_batch_sample_indices))])
        
        # updating probability of slices and their infections
        for bsi in range(len(current_batch_sample_indices)):

            ok_mask = (current_batch_slice_indices[bsi] >= 0)

            self.slices_probs[current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi][ok_mask]] = \
                slice_probs[bsi, :][ok_mask]
            self.slices_lobe_sizes[current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi][ok_mask]] = \
                samples_slices_info[bsi][ok_mask, 0]
            self.slices_assigned_inf[current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi][ok_mask]] = \
                samples_slices_info[bsi][ok_mask, 1]
            self.slices_model_inf[current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi][ok_mask]] = \
                samples_slices_info[bsi][ok_mask, 2]
            self.slices_union_inf[current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi][ok_mask]] = \
                samples_slices_info[bsi][ok_mask, 3]
            self.slices_model_fine_inf[current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi][ok_mask]] = \
                samples_slices_info[bsi][ok_mask, 4]
            self.sample_slice_thickness[current_batch_sample_indices[bsi]][current_batch_slice_indices[bsi][ok_mask]] = \
                (batch_lobe_thickness[bsi][ok_mask]).reshape((np.sum(ok_mask),))

            consolidated_lobes = np.sum(
                np.logical_not(ok_mask) &
                (0.25 <= batch_lobe_relative_locs[bsi]) &
                (batch_lobe_relative_locs <= 0.75))

            self.n_consolidated_middle_lobes[bsi] += consolidated_lobes

    def get_samples_results_summaries(self):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of strings (One string per sample containing the summaries related
        to the sample.) """
        softened_slices_probs_info = self.calculate_softened_probabilities()
        slices_paths = self.content_loader_of_interest.get_elements_names()

        def get_avg_top_perc_ratio(si):
            """ Returns the average infection ratio of top 20% slices based on model's assigned infection
            for the given sample index."""
            n_el = int(np.ceil(float(len(slices_paths[si])) / 5))
            return get_avg_top_k_ratio(si, n_el)

        def get_avg_top_k_ratio(si, n_el=4):
            """ Returns the average infection ratio of top n_el slices based on model's assigned infection
            for the given sample index."""
            model_infs = self.slices_model_inf[si]
            lobe_sizes = self.slices_lobe_sizes[si]

            n_el = min(len(model_infs), n_el)
            inds = np.argpartition(model_infs, -1 * n_el)[-1 * n_el:]
            return float(np.mean(
                model_infs[inds].astype(float) / lobe_sizes[inds]
            ))

        def get_avg_top_perc(si):
            """ Returns the average total infection of top 20% slices based on model's assigned infection
            for the given sample index."""
            n_el = int(np.ceil(float(len(slices_paths[si])) / 5))
            return get_avg_top_k(si, n_el)

        def get_avg_top_k(si, n_el=4):
            """ Returns the average total infection of top n_el slices based on model's assigned infection
            for the given sample index."""
            model_infs = self.slices_model_inf[si]

            n_el = min(len(model_infs), n_el)
            inds = np.argpartition(model_infs, -1 * n_el)[-1 * n_el:]
            return float(np.mean(
                model_infs[inds].astype(float)
            ))

        samples_all_masks_n = self.content_loader_of_interest.get_all_masks_num()

        def extract_info_for_one_sample(sid):
            """ Returns a str containing all the information for sample with the given ID"""

            softened_model_prob = np.amax(softened_slices_probs_info[sid][1])
            sample_slices = slices_paths[sid]
            left_lobe_mask = np.asarray(['left' in x for x in sample_slices])
            right_lobe_mask = np.asarray(['right' in x for x in sample_slices])

            left_lobe_size = np.sum(
                self.slices_lobe_sizes[sid][left_lobe_mask] *
                self.sample_slice_thickness[sid][left_lobe_mask], axis=0)
            right_lobe_size = np.sum(
                self.slices_lobe_sizes[sid][right_lobe_mask] *
                self.sample_slice_thickness[sid][right_lobe_mask], axis=0)

            left_lobe_area = np.sum(
                self.slices_lobe_sizes[sid][left_lobe_mask], axis=0)
            right_lobe_area = np.sum(
                self.slices_lobe_sizes[sid][right_lobe_mask], axis=0)

            left_lobe_model_inf = np.sum(self.slices_model_inf[sid][left_lobe_mask], axis=0)
            right_lobe_model_inf = np.sum(self.slices_model_inf[sid][right_lobe_mask], axis=0)

            left_lobe_model_fine_inf = np.sum(self.slices_model_fine_inf[sid][left_lobe_mask], axis=0)
            right_lobe_model_fine_inf = np.sum(self.slices_model_fine_inf[sid][right_lobe_mask], axis=0)

            left_lobe_assigned_inf = np.sum(self.slices_assigned_inf[sid][left_lobe_mask], axis=0)
            right_lobe_assigned_inf = np.sum(self.slices_assigned_inf[sid][right_lobe_mask], axis=0)

            left_lobe_union_inf = np.sum(self.slices_union_inf[sid][left_lobe_mask], axis=0)
            right_lobe_union_inf = np.sum(self.slices_union_inf[sid][right_lobe_mask], axis=0)

            def get_percentage(a, b):
                if b == 0:
                    return -1
                else:
                    return 100.0 * float(a) / b

            return '\t'.join([
                '%.2f' % (softened_model_prob,),
                '%.2f' % (left_lobe_size + right_lobe_size,),
                '%.0f' % (float(left_lobe_area + right_lobe_area) / samples_all_masks_n[sid],),
                '%.4f' % (get_percentage(left_lobe_model_inf + right_lobe_model_inf, left_lobe_area + right_lobe_area),),
                '%.4f' % (get_percentage(left_lobe_model_fine_inf + right_lobe_model_fine_inf, left_lobe_area + right_lobe_area),),
                '%.4f' % (get_percentage(left_lobe_assigned_inf + right_lobe_assigned_inf, left_lobe_area + right_lobe_area),),
                '%.4f' % (get_percentage(left_lobe_union_inf + right_lobe_union_inf, left_lobe_area + right_lobe_area),),
                '%.2f' % (left_lobe_size,),
                '%.4f' % (get_percentage(left_lobe_model_inf, left_lobe_area),),
                '%.4f' % (get_percentage(left_lobe_model_fine_inf, left_lobe_area),),
                '%.4f' % (get_percentage(left_lobe_assigned_inf, left_lobe_area),),
                '%.4f' % (get_percentage(left_lobe_union_inf, left_lobe_area),),
                '%.2f' % (right_lobe_size,),
                '%.4f' % (get_percentage(right_lobe_model_inf, right_lobe_area),),
                '%.4f' % (get_percentage(right_lobe_model_fine_inf, right_lobe_area),),
                '%.4f' % (get_percentage(right_lobe_assigned_inf, right_lobe_area),),
                '%.4f' % (get_percentage(right_lobe_union_inf, right_lobe_area),),
                '%.4f' % (get_avg_top_perc(sid),),
                '%.4f' % (get_avg_top_perc_ratio(sid),),
                '%.4f' % (get_avg_top_k(sid, 4),),
                '%.4f' % (get_avg_top_k_ratio(sid, 4),),
                '%.2f' % (self.n_consolidated_middle_lobes[sid] * self.sample_slice_thickness[sid][0],),
                '%.2f' % (samples_all_masks_n[sid] * self.sample_slice_thickness[sid][0],)
            ])

        return self.data_loader.get_samples_names(), [extract_info_for_one_sample(i)
                for i in range(len(slices_paths))]

    def get_samples_results_header(self):
        """ Returns the header of the results summaries saved for the samples in the report. """
        return ['PositiveClassProbability', 'LungVol',
                'NormLungSize', 'ModelInf%', 'ModelFineInf%', 'AssignedInf%', 'UnionInf%',
                'LLobeVol', 'LModelInf%', 'LModelFineInf%', 'LAssignedInf%', 'LUnionInf%',
                'RLobeVol', 'RModelInf%', 'RModelFineInf%', 'RAssignedInf%', 'RUnionInf%',
                'AvgTop20%Inf', 'AvgTop20%Inf%', 'AvgTop4Inf', 'AvgTop4Inf%',
                'NConsolidatedLobes*Thickness', 'NSlices*Thickness'
                ]

    def get_samples_elements_results_summaries(self):
        """ Receives as input a data loader containing information about the samples and a
        dictionary containing the variables, calculates summary for each sample, returns the
        summaries in a list of lists of values (one list per sample in which a value/numpy array 
        per element). """

        sample_slice_paths = self.content_loader_of_interest.get_elements_names()
        sample_slices_relative_positions = self.content_loader_of_interest.get_slices_relative_position()

        return [
            [
                (sample_slice_paths[si][ssi],
                 np.asarray([
                    sample_slices_relative_positions[si][ssi],
                    self.slices_probs[si][ssi],
                    self.slices_lobe_sizes[si][ssi] * self.sample_slice_thickness[si][ssi],
                    float(self.slices_model_inf[si][ssi]) / self.slices_lobe_sizes[si][ssi],
                    float(self.slices_assigned_inf[si][ssi]) / self.slices_lobe_sizes[si][ssi],
                    float(self.slices_union_inf[si][ssi]) / self.slices_lobe_sizes[si][ssi],
                    float(self.slices_model_fine_inf[si][ssi]) / self.slices_lobe_sizes[si][ssi]
                ]))
                for ssi in range(len(sample_slice_paths[si]))
            ]
            for si in range(len(sample_slice_paths))
        ]


def calculate_slices_info(args):
    """ Calculates summaries for the slices of one sample in the current batch"""
    
    inf_handler, bs_slice_indices, bs_prev_slice_sizes, bs_mask_labels, \
        bs_assigned_infs, bs_bbs, bs_patch_probs, org_imgs = args
    sample_info_summary = np.zeros((len(bs_slice_indices), 5), dtype=float)

    for bssi in range(len(bs_slice_indices)):

        # for each slice, if lobe size is 0 (not added before) do all the heavy computations
        #if bs_prev_slice_sizes[bssi] > 0:
            #continue

        lobe_cropped_mask = inf_handler.ih.crop_bounding_box(
            bs_mask_labels[bssi, :, :], bs_bbs[bssi, :])

        lobe_size = np.sum(lobe_cropped_mask)

        assigned_cropped_inf_mask = \
            (inf_handler.ih.get_original_mask_in_bounding_box(
                bs_bbs[bssi, :], bs_assigned_infs[bssi, :, :]
            ) >= 0.5).astype(int) * lobe_cropped_mask

        assigned_inf = np.sum(assigned_cropped_inf_mask)

        model_cropped_inf_mask = (inf_handler.ih.get_original_mask_in_bounding_box(
            bs_bbs[bssi, :],
            inf_handler.ih.get_pixels_masks_from_neuron_masks(bs_patch_probs[bssi, :, :])
        ) >= 0.5).astype(int) * lobe_cropped_mask

        org_cropped_img = inf_handler.ih.get_original_mask_in_bounding_box(
                bs_bbs[bssi, :], org_imgs[bssi]
            )

        model_fine_cropped_inf_mask = inner_infection(
            org_cropped_img
            , assigned_cropped_inf_mask, model_cropped_inf_mask)
        fine_inf = np.sum(model_fine_cropped_inf_mask)

        model_inf = np.sum(model_cropped_inf_mask)

        union_inf_mask = ((model_cropped_inf_mask + assigned_cropped_inf_mask) >= 1).astype(int)
        union_inf = np.sum(union_inf_mask)

        sample_info_summary[bssi] = \
            np.asarray([lobe_size, assigned_inf, model_inf, union_inf, fine_inf])

    return sample_info_summary


class TopLevelInfHandler:

    def __init__(self, conf):
        self.ih = conf['patch_labler']


def inner_infection(img, assigned_inf_mask, model_mask):
    model_msk = model_mask.astype(int)
    dilation = morphology.dilation(model_msk, np.ones([8, 8]))
    msk_raw = morphology.erosion(dilation, np.ones([8, 8]))

    # Inside Infection
    mask_values = img[model_msk]
    model_msk = model_msk.astype(int)

    if len(mask_values) > 0:
        med = np.median(mask_values)
        mean = np.mean(mask_values)
        std = np.std(mask_values)

        if med > std:
            thr2 = med
        elif mean > std:
            thr2 = mean
        else:
            if med < mean:
                thr2 = med + std
            else:
                thr2 = mean + std
        thr1 = np.minimum(std, 11000)
        thr1 = np.maximum(thr1, 8000)
        thr2 = np.maximum(thr2, 25000)

        inside_infection_mask = (thr1 < img) * (img < thr2)
        inside_infection = inside_infection_mask * model_msk
        cus_msk = np.bitwise_or(assigned_inf_mask > 0, inside_infection > 0)
    else:
        cus_msk = msk_raw > 0
    return np.bitwise_and(cus_msk, model_mask)
