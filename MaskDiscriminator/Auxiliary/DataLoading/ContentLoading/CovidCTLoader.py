from os import listdir, path
import numpy as np
import re as regex
from Auxiliary.DataLoading.ContentLoading.ContentLoader import ContentLoader
from Auxiliary.Threading.WorkerCoordinating import WorkersCoordinator
import pandas as pd


class CovidCTLoader(ContentLoader):

    def __init__(self, conf, prefix_name, data_specification,
                 slices_information_dir=None, single_sample=False,
                 table_information_dir=None):
        super(CovidCTLoader, self).__init__(conf, prefix_name, data_specification)

        self.train_phase = conf['phase'] == 'train'

        self.samples_slices_paths = []
        self.samples_paths = []
        self.samples_labels = []

        # Loading samples
        self.load_samples(data_specification, single_sample)
        
        # Saving information in order to prevent rebuilding them!
        self.unique_base_names = None
        self.unique_base_names_views_indices = None
        
        self.check_data_completeness()

        # For mapping lobes and slices to each other
        self.sample_slices_left_lobe_indices = None
        self.sample_slices_right_lobe_indices = None
        self.lobes_whole_slice_index = None

        self.separate_data_masks_and_related_lobes()

        self.slices_information_dir = slices_information_dir

        self.sample_heights = self.set_samples_heights()

        self.loader_workers = WorkersCoordinator(4)

        self.slices_probs = None  # keeping data to load it fasttt

        self.table_info = None
        if table_information_dir is not None:
            self.table_info = pd.read_csv(table_information_dir, sep='\t', header=0).values
            # sorting by names to lookup easier!
            self.table_info = self.table_info[np.argsort(self.table_info[:, 0]), :]

    def validate(self):

        cnt = 0

        for i in range(len(self.samples_paths)):

            if len(self.lobes_whole_slice_index[i]) != len(self.samples_slices_paths[i]):
                print('Mismatched slice lengths!!!')
                cnt += 1

            if (np.amax(self.sample_slices_left_lobe_indices[i]) >= len(self.samples_slices_paths[i])) or \
                (np.amax(self.sample_slices_right_lobe_indices[i]) >= len(self.samples_slices_paths[i])):
                print('Index of left/right slice more than the number of slices')
                cnt += 1

            if np.any(np.vectorize(lambda x: 'right' in x)(np.asarray(self.samples_slices_paths[i])[self.sample_slices_left_lobe_indices[i][self.sample_slices_left_lobe_indices[i] > -1]])):
                print('right path in left slices group!!!')
                cnt += 1

            if np.any(np.vectorize(lambda x: 'left' in x)(np.asarray(self.samples_slices_paths[i])[self.sample_slices_right_lobe_indices[i][self.sample_slices_right_lobe_indices[i] > -1]])):
                print('left path in right slices group!!!')
                cnt += 1

        print('%d problems found' % cnt)
        if cnt > 0:
            raise Exception('Problems found!!!')

    def loads_multi_view_loader(self):
        return True

    def load_samples(self, samples_specifications, single_sample):

        samples_specifications_copy = np.copy(samples_specifications)
        name_filter = None

        if single_sample:
            data_paths = [samples_specifications]

        else:
            if ':' in samples_specifications:
                name_filter = '/' + samples_specifications.split(':')[1] + '/'
                samples_specifications = samples_specifications.split(':')[0]

            if samples_specifications in ['train', 'val', 'test']:
                samples_specifications = '../DataSeparation/%s/%s.txt' % \
                                         (self.conf['dataSeparation'], samples_specifications)

            if path.isfile(samples_specifications):
                data_paths = []

                with open(samples_specifications, 'r') as f:
                    for fl in f:
                        fl = fl.strip()
                        if fl == '':
                            continue
                        l_tag, l_path = fl.split(':')
                        data_paths.append((int(l_tag), l_path))
            else:
                data_paths = discover_samples(samples_specifications)

        # doing some process, filtering slices with no clear cut of lung
        skip_count = 0
        n_samples_per_class = dict()

        for sample_label, sample_path in data_paths:

            if not path.exists(sample_path):
                print('%s has been removed!!!' % sample_path)
                skip_count += 1
                continue

            if len(listdir(sample_path)) > 0:
                good_slices = find_good_slices(sample_path)
                if len(good_slices) == 0:
                    skip_count += 1
                    continue
                if name_filter is not None and name_filter not in sample_path:
                    skip_count += 1
                    continue

                # Checking if samples with the label similar to the sample should be considered in loader or skipped
                if sample_label not in self.conf['LabelMapDict']:
                    skip_count += 1
                    continue

                # Mapping the label to the label defined by model's configurations
                # (e.g. for grouping different types of labels)
                mapped_label = self.conf['LabelMapDict'][sample_label]

                # if debug mode, clipping slices and keeping 9 slices
                if self.conf['debug_mode'] and len(good_slices) > 10:
                    print('@@@ Clipping slices of the samples to atmost 10 slices in the debugging mode.')
                    good_slices = good_slices[:min(len(good_slices), np.random.randint(1, 10))]

                self.samples_slices_paths.append(good_slices)
                self.samples_labels.append(mapped_label)
                self.samples_paths.append(sample_path)

                if mapped_label not in n_samples_per_class:
                    n_samples_per_class[mapped_label] = 0
                n_samples_per_class[mapped_label] += 1

        self.samples_labels = np.asarray(self.samples_labels)

        print('%d skips because of name/conditions!' % skip_count)
        print(', '.join(['%d class-%d' % (n_samples_per_class[class_label], class_label)
                         for class_label in sorted(n_samples_per_class.keys())]) +
              ' samples/views found for data: %s' % samples_specifications_copy)

    def check_data_completeness(self):
        """ Checks the required extra files for all the data, raises exception if they are not available
        before beginning the run so nothing will be messed in the middle! """

        def check_existence_of_patch_labels():
            """ Checks the existence of all patch infection labels files of the given samples """

            # counting the number of slices with problems
            problem_cnt = 0

            for all_slices in self.samples_slices_paths:
                for sp in all_slices:
                    if 'patch_labels_dir' in self.conf:
                        if not path.exists(sp.replace('..', self.conf['patch_labels_dir'])):
                            print('No path ' + sp.replace('..', self.conf['patch_labels_dir']))
                            problem_cnt += 1

                    elif not path.exists(self.conf['patch_labler'].get_mask_dir_for_file(sp)):
                        print('No path ' + self.conf['patch_labler'].get_mask_dir_for_file(sp))
                        problem_cnt += 1

            if problem_cnt > 0:
                raise Exception('Missing patch infection labels in the mentioned samples')

        def check_existence_of_patch_dists():
            """ Checks the existence of all patch distance files of the given samples """

            # counting the number of slices with problems
            problem_cnt = 0

            for all_slices in self.samples_slices_paths:
                for sp in all_slices:
                    if not path.exists(self.conf['distMaskDirGiver'](sp)):
                        print('No path ' + self.conf['distMaskDirGiver'](sp))
                        problem_cnt += 1

            if problem_cnt > 0:
                raise Exception('Missing patch infection labels in the mentioned samples')

        def check_existence_of_slice_probs():
            """ Checks the existence of all patch distance files of the given samples """

            addr_giver = (lambda slice_path: self.conf['slice_probs_dir'] + slice_path.replace('..', ''))

            # counting the number of slices with problems
            problem_cnt = 0

            for all_slices in self.samples_slices_paths:
                for sp in all_slices:
                    if not path.exists(addr_giver(sp)):
                        print('No path ' + addr_giver(sp))
                        problem_cnt += 1

            if problem_cnt > 0:
                raise Exception('Missing patch infection labels in the mentioned samples')

        def check_existence_of_patch_probs():
            """ Checks the existence of all patch distance files of the given samples """

            addr_giver = (lambda slice_path: self.conf['patch_probs_dir'] + slice_path.replace('..', ''))

            # counting the number of slices with problems
            problem_cnt = 0

            for all_slices in self.samples_slices_paths:
                for sp in all_slices:
                    if not path.exists(addr_giver(sp)):
                        print('No path ' + addr_giver(sp))
                        problem_cnt += 1

            if problem_cnt > 0:
                raise Exception('Missing patch infection labels in the mentioned samples')

        # patch labels
        if self.conf['patch_labler'] is not None and self.conf['phase'] == 'train':
            check_existence_of_patch_labels()

        # patch distances
        if self.conf['distMaskDirGiver'] is not None:
            check_existence_of_patch_dists()

        if 'slice_probs_dir' in self.conf and self.conf['slice_probs_dir'] is not None and \
                'skip_checking_slice_probs' not in self.conf:
            check_existence_of_slice_probs()

        if 'patch_probs_dir' in self.conf and self.conf['patch_probs_dir'] is not None:
            check_existence_of_patch_probs()

    def separate_data_masks_and_related_lobes(self):

        self.sample_slices_left_lobe_indices = []
        self.sample_slices_right_lobe_indices = []
        self.lobes_whole_slice_index = []

        for i in range(len(self.samples_paths)):
            slices_heights = np.vectorize(lambda fd: float(path.basename(fd).split('_')[1]))(self.samples_slices_paths[i])
            unique_heights = np.unique(slices_heights)

            lobes_whole_slice_index = np.searchsorted(unique_heights, slices_heights, side='left')
            left_mask = ['left' in x for x in self.samples_slices_paths[i]]
            right_mask = ['right' in x for x in self.samples_slices_paths[i]]
            lobe_indices = np.arange(len(left_mask))
            left_lobe_indices = lobe_indices[left_mask]
            right_lobe_indices = lobe_indices[right_mask]

            self.lobes_whole_slice_index.append(lobes_whole_slice_index)

            slices_left_lobes = np.full((len(unique_heights),), -1, dtype=int)
            slices_left_lobes[lobes_whole_slice_index[left_mask]] = left_lobe_indices
            slices_right_lobes = np.full((len(unique_heights),), -1, dtype=int)
            slices_right_lobes[lobes_whole_slice_index[right_mask]] = right_lobe_indices

            s1 = np.sum(np.asarray(['right' in self.samples_slices_paths[i][x]
                                   for x in slices_left_lobes if x != -1]).astype(int))
            s2 = np.sum(np.asarray(['left' in self.samples_slices_paths[i][x]
                                   for x in slices_right_lobes if x != -1]).astype(int))
            if s1 >0 or s2 > 0:
                print('Problem: ', s1, s2)

            self.sample_slices_left_lobe_indices.append(slices_left_lobes)
            self.sample_slices_right_lobe_indices.append(slices_right_lobes)

    def get_samples_names(self):
        return self.samples_paths

    def get_samples_labels(self):
        return self.samples_labels

    def get_elements_names(self):
        """ Returns a list of lists, one list per sample containing name of its elements. """
        return self.samples_slices_paths

    def set_samples_heights(self):
        get_height = np.vectorize(lambda fd: float(path.basename(fd).split('_')[1]))

        def get_sample_max_height(sample_path):
            mask_slices = np.asarray([x for x in listdir(sample_path) if '_mask.' in x])
            slice_heights = get_height(mask_slices)
            return np.amax(slice_heights)

        samples_heights = np.asarray([get_sample_max_height(x) for x in self.samples_paths])
        return samples_heights

    def get_samples_batch_effect_groups(self):
        """ Returns a dictionary from each class label to one list per class label.
        The list contains lists of indices of the samples related to one batch effect group, e.g.
        the ones captured in one hospital!"""

    def reorder_samples(self, reorder_inds, new_samples_names):

        self.samples_paths = [self.samples_paths[i] for i in reorder_inds]
        self.samples_slices_paths = [self.samples_slices_paths[i] for i in reorder_inds]
        self.samples_labels = self.samples_labels[reorder_inds]

        self.unique_base_names = None
        self.unique_base_names_views_indices = None

        self.lobes_whole_slice_index = [self.lobes_whole_slice_index[i] for i in reorder_inds]
        self.sample_slices_left_lobe_indices = [self.sample_slices_left_lobe_indices[i] for i in reorder_inds]
        self.sample_slices_right_lobe_indices = [self.sample_slices_right_lobe_indices[i] for i in reorder_inds]

        self.sample_heights = self.sample_heights[reorder_inds]

    def get_views_indices(self):

        if self.unique_base_names is not None and \
                self.unique_base_names_views_indices is not None:
            print(self.unique_base_names)
            print(self.unique_base_names_views_indices)
            return self.unique_base_names, self.unique_base_names_views_indices
        
        # Separating base sample name
        def get_base_sample_name(p):
            if 'PreprocessedData' in p:
                return p[:p.rfind('_')]
            else:
                return p

        v_get_base_sample_name = np.vectorize(get_base_sample_name)
        base_names = v_get_base_sample_name(np.asarray(self.samples_paths))
        u_base_names = np.unique(base_names)
        print('%d unique samples out of %d total views.' % (len(u_base_names), len(self.samples_paths)))
        print('%d class-0, %d class-1 and %d class-2 unique samples found.' %
              tuple([len([x for x in u_base_names if '/%d/' % l in x]) for l in range(3)]))

        # finding the index of the base name for each sample
        base_indices = np.searchsorted(u_base_names, base_names, side='left')

        u_base_names = list(u_base_names)
        base_names_view_indices = [[] for _ in range(len(u_base_names))]
        for i in range(len(base_indices)):
            base_names_view_indices[base_indices[i]].append(i)

        # Saving information
        self.unique_base_names = u_base_names
        self.unique_base_names_views_indices = base_names_view_indices
        
        return u_base_names, base_names_view_indices

    def get_placeholder_name_to_fill_function_dict(self):
        return {
            'images': self.prepare_batch_preprocessed_images,
            'images_left': (lambda x: self.prepare_batch_preprocessed_images(x, 'left')),
            'images_right': (lambda x: self.prepare_batch_preprocessed_images(x, 'right')),
            'labels': self.prepare_batch_labels,
            'images_patches_infection_labels': self.prepare_batch_patch_infection_labels,
            'images_patches_distances': self.prepare_batch_patch_dists,
            'images_patches_distances_left': (lambda x: self.prepare_batch_patch_dists(x, 'left')),
            'images_patches_distances_right': (lambda x: self.prepare_batch_patch_dists(x, 'right')),
            'elements_info': self.prepare_batch_elements_info,
            'left_right_lobes_info': self.prepare_batch_left_right_lobes_info,
            'thickness': self.prepare_batch_thickness,
            'slice_right_probs': (lambda x: self.prepare_batch_slice_probs(x, 'right')),
            'slice_left_probs': (lambda x: self.prepare_batch_slice_probs(x, 'left')),
            'patch_right_probs': (lambda x: self.prepare_batch_patch_probs(x, 'right')),
            'patch_left_probs': (lambda x: self.prepare_batch_patch_probs(x, 'left')),
            'slice_thickness': self.prepare_batch_slice_thickness,
            'slice_right_thickness': (lambda x: self.prepare_batch_slice_thickness(x, 'right')),
            'slice_left_thickness': (lambda x: self.prepare_batch_slice_thickness(x, 'left')),
            'slice_relative_location': self.prepare_batch_slice_relative_location,
            'slice_right_relative_location': (lambda x: self.prepare_batch_slice_relative_location(x, 'right')),
            'slice_left_relative_location': (lambda x: self.prepare_batch_slice_relative_location(x, 'left')),
            'slice_missing_mask': self.prepare_batch_slice_missing_mask,
            'left_slice_missing_mask': (lambda x: self.prepare_batch_slice_missing_mask(x, 'left')),
            'right_slice_missing_mask': (lambda x: self.prepare_batch_slice_missing_mask(x, 'right')),
            'sample_table_info': self.prepare_batch_samples_table_info,
        }

    def get_left_and_right_indices_for_batch(self, sample_inds, elements_inds):
        samples_unique_masks_ids = [
            np.unique([(self.lobes_whole_slice_index[sample_inds[i]])[ssi]
                       for ssi in elements_inds[i] if ssi != -1])
            for i in range(len(sample_inds))]

        sample_masks_n = [len(x) for x in samples_unique_masks_ids]
        max_len = max(sample_masks_n)

        batch_left_lobes_inds = [
            np.concatenate((
                self.sample_slices_left_lobe_indices[sample_inds[i]][samples_unique_masks_ids[i]],
                np.full((max_len - sample_masks_n[i],), -1, dtype=int)
            ), axis=0)
            for i in range(len(samples_unique_masks_ids))]

        batch_right_lobes_inds = [
            np.concatenate((
                self.sample_slices_right_lobe_indices[sample_inds[i]][samples_unique_masks_ids[i]],
                np.full((max_len - sample_masks_n[i],), -1, dtype=int)
            ), axis=0)
            for i in range(len(samples_unique_masks_ids))]

        return batch_left_lobes_inds, batch_right_lobes_inds

    def stack_batch_samples(self, batch_chooser, get_one_slice_result, lobe=None):
        sample_inds = batch_chooser.get_current_batch_sample_indices()
        elements_inds = batch_chooser.get_current_batch_elements_indices()

        # if lobe is not None, selecting from unique masks
        if lobe is not None:
            samples_unique_masks_ids = [
                np.unique([(self.lobes_whole_slice_index[sample_inds[i]])[ssi]
                           for ssi in elements_inds[i] if ssi != -1])
                for i in range(len(sample_inds))]

            sample_masks_n = [len(x) for x in samples_unique_masks_ids]
            max_len = max(sample_masks_n)

            if lobe == 'left':
                batch_left_lobes_inds = [
                    np.concatenate((
                        self.sample_slices_left_lobe_indices[sample_inds[i]][samples_unique_masks_ids[i]],
                        np.full((max_len - sample_masks_n[i],), -1, dtype=int)
                    ), axis=0)
                    for i in range(len(samples_unique_masks_ids))]
                elements_inds = batch_left_lobes_inds

            if lobe == 'right':
                batch_right_lobes_inds = [
                    np.concatenate((
                        self.sample_slices_right_lobe_indices[sample_inds[i]][samples_unique_masks_ids[i]],
                        np.full((max_len - sample_masks_n[i],), -1, dtype=int)
                    ), axis=0)
                    for i in range(len(samples_unique_masks_ids))]
                elements_inds = batch_right_lobes_inds

        results = self.loader_workers.run_func(get_one_slice_result,
                                               [(sample_inds[si], ssi)
                                                for si in range(len(sample_inds))
                                                for ssi in elements_inds[si]])

        '''
        return np.stack(tuple([
            np.stack(tuple([
                get_one_slice_result(sample_inds[si], ssi)
                for ssi in elements_inds[si]
            ]), axis=0)
            for si in range(len(sample_inds))
        ]), axis=0)
        '''
        ret = np.stack(tuple([
            np.stack(tuple([
                results[si * len(elements_inds[0]) + ssi]
                for ssi in range(len(elements_inds[si]))
            ]), axis=0)
            for si in range(len(sample_inds))
        ]), axis=0)

        return ret

    def prepare_batch_preprocessed_images(self, batch_chooser, lobe=None):
        """ Prepares and returns preprocessed images of the specified batch in
        batch chooser. """

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros((3, 256, 256), dtype=float)
            else:
                return prepare_sample_per_slice(
                    self.samples_slices_paths[sample_i], slice_i, self)

        return self.stack_batch_samples(batch_chooser, get_one_slice_result, lobe)

    def prepare_batch_labels(self, batch_chooser):
        
        sample_inds = batch_chooser.get_current_batch_sample_indices()

        return self.samples_labels[sample_inds]

    def prepare_batch_patch_infection_labels(self, batch_chooser):
        """ Prepares and returns infection labels for the current batch of the batch chooser """

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros((self.conf['patch_labler'].neuron_cnt, self.conf['patch_labler'].neuron_cnt), dtype=float)
            elif 'patch_labels_dir' in self.conf:
                return (np.load(self.samples_slices_paths[sample_i][slice_i].
                               replace('..', self.conf['patch_labels_dir'])) >= 0.5).astype(np.float32)

            else:
                return np.load(self.conf['patch_labler'].get_mask_dir_for_file(
                        self.samples_slices_paths[sample_i][slice_i])).astype(np.float32)

        return self.stack_batch_samples(batch_chooser, get_one_slice_result)

    def prepare_batch_pixel_dists(self, batch_chooser, lobe=None):
        """ Prepares and returns (distances of patches from lobe peripheral)
         for the current specified batch in batch chooser """

        addr_giver = (lambda x: x.replace('left', 'dist_left').replace('right', 'dist_right'))

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros((self.conf['patches_cnt_in_row'], self.conf['patches_cnt_in_row']),
                                dtype=float)
            else:
                return np.load(addr_giver(
                    self.samples_slices_paths[sample_i][slice_i])).astype(np.float32)

        return self.stack_batch_samples(batch_chooser, get_one_slice_result, lobe)

    def prepare_batch_patch_dists(self, batch_chooser, lobe=None):
        """ Prepares and returns (distances of patches from lobe peripheral)
         for the current specified batch in batch chooser """

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros((self.conf['patches_cnt_in_row'], self.conf['patches_cnt_in_row']),
                                dtype=float)
            else:
                return np.load(self.conf['distMaskDirGiver'](
                    self.samples_slices_paths[sample_i][slice_i])).astype(np.float32)

        return self.stack_batch_samples(batch_chooser, get_one_slice_result, lobe)

    def prepare_batch_mask_labels(self, batch_chooser):
        """ Prepares and returns the whole slice mask label (512 x 512)
        for the current specified batch"""

        get_mask_path = (lambda slice_path: slice_path.replace('left', 'mask').replace('right', 'mask'))

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros((512, 512), dtype=int)
            else:
                return np.load(get_mask_path(
                    self.samples_slices_paths[sample_i][slice_i])).astype(int)

        return (self.stack_batch_samples(batch_chooser, get_one_slice_result) > 0).astype(int)

    def prepare_batch_preprocessed_untouched_images(self, batch_chooser):
        """ Prepares and returns the the original preprocessed
        for the current specified batch (saved sample_indices and slice_indices) """

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros((256, 256), dtype=int)
            else:
                return np.load(
                    self.samples_slices_paths[sample_i][slice_i]).astype(int)

        return self.stack_batch_samples(batch_chooser, get_one_slice_result)

    def prepare_batch_pixel_level_inf_labels(self, batch_chooser):
        """ Prepares and returns the assigned infection at pixel resolution (256 x 256)
        for the current specified batch (saved sample_indices and slice_indices) """

        get_mask_path = (lambda slice_path: slice_path.replace('left', 'inf_left').replace('right', 'inf_right'))

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros((256, 256), dtype=float)
            else:
                return np.load(get_mask_path(
                    self.samples_slices_paths[sample_i][slice_i]))

        return self.stack_batch_samples(batch_chooser, get_one_slice_result)

    def prepare_batch_bounding_boxes(self, batch_chooser):
        """ Prepares and returns the bounding box of the lobes (4)
        for the current specified batch (saved sample_indices and slice_indices) """

        get_bb_path = (lambda slice_path: slice_path.replace('left', 'bd_left').replace('right', 'bd_right'))

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros((4,), dtype=int)
            else:
                return np.load(get_bb_path(
                    self.samples_slices_paths[sample_i][slice_i])).astype(int)

        return self.stack_batch_samples(batch_chooser, get_one_slice_result)

    def prepare_batch_elements_info(self, batch_chooser):

        get_info_path = (lambda save_dir, slice_path: save_dir + '/' + '/'.join(slice_path.split('/')[2:]))

        typical_shape = (np.load(get_info_path(
                    self.slices_information_dir,
                    self.samples_slices_paths[0][0])).astype(np.float32)).shape

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros(typical_shape, dtype=float)
            else:
                return np.load(get_info_path(
                    self.slices_information_dir,
                    self.samples_slices_paths[sample_i][slice_i])).astype(np.float32)

        return self.stack_batch_samples(batch_chooser, get_one_slice_result)

    def prepare_batch_left_right_lobes_info(self, batch_chooser):

        get_info_path = (lambda slice_path: self.slices_information_dir + '/' + '/'.join(slice_path.split('/')[2:]))

        sample_inds = batch_chooser.get_current_batch_sample_indices()
        elements_inds = batch_chooser.get_current_batch_elements_indices()

        samples_unique_masks_ids = [
            np.unique([(self.lobes_whole_slice_index[sample_inds[i]])[ssi] 
                       for ssi in elements_inds[i] if ssi != -1]) 
            for i in range(len(sample_inds))]

        sample_masks_n = [len(x) for x in samples_unique_masks_ids]
        max_len = max(sample_masks_n)

        batch_left_lobes_inds = [
            np.concatenate((
                (self.sample_slices_left_lobe_indices[sample_inds[i]])[samples_unique_masks_ids[i]],
                np.full((max_len - sample_masks_n[i],), -1, dtype=int)
            ), axis=0) 
            for i in range(len(samples_unique_masks_ids))]

        batch_right_lobes_inds = [
            np.concatenate((
                (self.sample_slices_right_lobe_indices[sample_inds[i]])[samples_unique_masks_ids[i]],
                np.full((max_len - sample_masks_n[i],), -1, dtype=int)
            ), axis=0)
            for i in range(len(samples_unique_masks_ids))]

        typical_shape = (np.load(
            self.slices_information_dir + '/' + '/'.join(
                self.samples_slices_paths[0][0].split('/')[2:]))).shape
        
        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros(typical_shape, dtype=float)
            else:
                return np.load(get_info_path(
                    self.samples_slices_paths[sample_i][slice_i])).astype(np.float32)

        results = self.loader_workers.run_func(get_one_slice_result,
                                               [(sample_inds[si], ssi)
                                                for si in range(len(sample_inds))
                                                for ssi in batch_left_lobes_inds[si]])

        left_info = np.stack(tuple([
            np.stack(tuple([
                results[si * len(batch_left_lobes_inds[0]) + ssi]
                for ssi in range(len(batch_left_lobes_inds[0]))
            ]), axis=0)
            for si in range(len(sample_inds))
        ]), axis=0)

        results = self.loader_workers.run_func(get_one_slice_result,
                                               [(sample_inds[si], ssi)
                                                for si in range(len(sample_inds))
                                                for ssi in batch_right_lobes_inds[si]])

        right_info = np.stack(tuple([
            np.stack(tuple([
                results[si * len(batch_right_lobes_inds[0]) + ssi]
                for ssi in range(len(batch_right_lobes_inds[0]))
            ]), axis=0)
            for si in range(len(sample_inds))
        ]), axis=0)

        return np.concatenate((left_info, right_info), axis=-1)

    def prepare_batch_thickness(self, batch_chooser):

        sample_inds = batch_chooser.get_current_batch_sample_indices()
        thickness_giver = np.vectorize(lambda si: float(path.basename(self.samples_slices_paths[si][0]).split('_')[2]))

        return thickness_giver(sample_inds)

    def load_slice_probs(self, the_dir):
        addr_giver = (lambda slice_path: the_dir + slice_path.replace('..', ''))

        print('Loading the probabilities for the whole data, this may take several minutes!!')
        all_slices_paths = [(ssp, 0) for sp in self.samples_slices_paths for ssp in sp]
        all_slices_probs = np.asarray(self.loader_workers.run_func(
            (lambda x, y: float(np.load(addr_giver(x)))), all_slices_paths))

        cursor = 0
        slices_probs = [[] for _ in range(len(self.samples_paths))]
        for i in range(len(self.samples_slices_paths)):
            slices_probs[i] = all_slices_probs[cursor: cursor + len(self.samples_slices_paths[i])]
            cursor += len(self.samples_slices_paths[i])

        return slices_probs

    def prepare_batch_slice_probs(self, batch_chooser, lobe=None):
        """ Prepares and returns preprocessed images of the specified batch in
        batch chooser. """

        if self.slices_probs is None:
            self.slices_probs = self.load_slice_probs(self.conf['slice_probs_dir'])

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros(self.slices_probs[0][0].shape, dtype=float)
            else:
                return self.slices_probs[sample_i][slice_i]

        return self.stack_batch_samples(batch_chooser, get_one_slice_result, lobe)

    def prepare_batch_patch_probs(self, batch_chooser, lobe=None):
        """ Prepares and returns preprocessed images of the specified batch in
        batch chooser. """

        addr_giver = (lambda slice_path: self.conf['patch_probs_dir'] + slice_path.replace('..', ''))

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.zeros(np.load(addr_giver(self.samples_slices_paths[0][0])).shape, dtype=float)
            else:
                return np.load(addr_giver(self.samples_slices_paths[sample_i][slice_i]))

        return self.stack_batch_samples(batch_chooser, get_one_slice_result, lobe)

    def prepare_batch_slice_thickness(self, batch_chooser, lobe=None):
        """ Prepares and returns preprocessed images of the specified batch in
        batch chooser. """

        thickness_giver = np.vectorize(lambda slice_path: float(path.basename(slice_path).split('_')[2]))

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.asarray([0])
            else:
                return np.asarray([thickness_giver(self.samples_slices_paths[sample_i][slice_i])])

        return self.stack_batch_samples(batch_chooser, get_one_slice_result, lobe)

    def prepare_batch_slice_relative_location(self, batch_chooser, lobe=None):
        """ Prepares and returns preprocessed images of the specified batch in
        batch chooser. """

        get_height = np.vectorize(lambda fd: float(path.basename(fd).split('_')[1]))

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.asarray([0])
            else:
                return np.asarray([1.0 * get_height(self.samples_slices_paths[sample_i][slice_i]) / self.sample_heights[sample_i]])

        return self.stack_batch_samples(batch_chooser, get_one_slice_result, lobe)

    def prepare_batch_slice_missing_mask(self, batch_chooser, lobe=None):
        """ Prepares and returns preprocessed images of the specified batch in
        batch chooser. """

        def get_one_slice_result(sample_i, slice_i):
            if slice_i == -1:
                return np.asarray([1])
            else:
                return np.asarray([0])

        return self.stack_batch_samples(batch_chooser, get_one_slice_result, lobe)

    def get_all_slices_paths(self):
        """ Returns a list containing the list of paths for slices of the samples
         (all the slices, without size limit)"""
        return [[s + '/' + ss
                 for ss in listdir(s) if
                 ('left' in ss or 'right' in ss) and
                 ('_bd_' not in ss and '_inf_' not in ss and '_dist_' not in ss)
        ]
            for s in self.samples_paths]

    def get_all_masks_num(self):
        """ Returns the number of all masks, an indicator of the number of slices per lung
        which is used for normalized lung size calculation"""
        return [len([s + '/' + ss
                 for ss in listdir(s) if
                 'mask' in ss and ('_inf_' not in ss and '_dist_' not in ss)
                 ])
                for s in self.samples_paths]

    def get_slices_relative_position(self):
        """ Returns the ratio of the height of one slice and the max height of the lung!"""

        get_height = np.vectorize(lambda fd: float(path.basename(fd).split('_')[1]))

        return [
            [
                1.0 * get_height(self.samples_slices_paths[si][ssi]) / self.sample_heights[si]
                for ssi in range(len(self.samples_slices_paths[si]))
            ]
            for si in range(len(self.samples_slices_paths))
        ]

    def prepare_batch_samples_table_info(self, batch_chooser):

        sample_inds = batch_chooser.get_current_batch_sample_indices()

        batch_samples = np.asarray([self.samples_paths[i] for i in sample_inds])

        return self.table_info[
               np.searchsorted(self.table_info[:, 0], batch_samples, side='left'), 1:].astype(float)


def read_single_img(img_dir, augmenter=None):
    """ Reads the image from the given path, normalizes it (sets the values between 0 and 1),
    applies augmenter to it (if not None) and returns the result matrix. """
    try:
        img = np.load(img_dir)
    except:
        print('Problem with ', img_dir)
        return None

    img = (img.astype(np.float64) / pow(2, 16)).astype(np.float32)
    if augmenter is not None:
        img = augmenter.augment_image(img)

    return img


def discover_samples(samples_dir):
    """ Looks for samples in all possible depths of the given directory: samples_dirs.
    The given directory must have at least one folder (per data group/hospital/...) and each group
    must have subdirectories dividing their samples based on the labels of the samples
    (each subdirectory is translated as the label of all the samples inside it and must be an integer).
    Returns a flat list containing tuples of (SampleLabel, SampleDir)s."""

    print('Discovering samples in ',  samples_dir)

    samples = []

    dirs_to_search = [(0, samples_dir + '/0'), (1, samples_dir + '/1'), (2, samples_dir + '/2')]

    # for unknown samples
    for f in listdir(samples_dir):
        if f not in ['0', '1', '2']:
            dirs_to_search.append((3, samples_dir + '/' + f))

    total_data_count = [0 for _ in range(4)]

    # for checking if two levels below or ct files
    def chech_if_sample_container(the_dir):
        dirs_subs = listdir(the_dir)
        if ".DS_Store" in dirs_subs:
            dirs_subs.remove(".DS_Store")

        if len(dirs_subs) == 0:
            return False

        for fsd in dirs_subs:
            sub_subs = listdir(the_dir + '/' + fsd)
            if ".DS_Store" in sub_subs:
                sub_subs.remove(".DS_Store")
            if len(sub_subs) > 0:
                if path.isfile('%s/%s/%s' % (the_dir, fsd, sub_subs[0])):
                    return True
                else:
                    return False

        return False

    while len(dirs_to_search) > 0:
        next_label, next_dir = dirs_to_search.pop(0)

        if ".DS_Store" in next_dir:
            continue

        if not path.exists(next_dir):
            continue

        subdirs = listdir(next_dir)
        if ".DS_Store" in subdirs:
            subdirs.remove(".DS_Store")

        if len(subdirs) == 0:
            continue

        if not chech_if_sample_container(next_dir):
            dirs_to_search += [(next_label, next_dir + '/' + nsd) for nsd in subdirs]

        else:
            total_data_count[next_label] += len(subdirs)
            samples += [(next_label, next_dir + '/' + sd) for sd in subdirs]

    print('Total samples: %d healthy, %d coronoid, %d abnormal, %d unknown' %
          tuple(total_data_count))
    return samples


def get_prev_slice_index(all_slices, slice_index):
    """ Returns the index of the upper slice of the given slice_index that is in the same lobe as the given index in the
    array of all slices. Returns the same slice if no upper slice in the lobe exist."""

    lobe_side = 'left'
    if 'right' in all_slices[slice_index]:
        lobe_side = 'right'

    i = slice_index - 1
    while i >= 0 and lobe_side not in all_slices[i]:
        i -= 1

    if i == -1:
        return slice_index
    else:
        return i


def get_next_slice_index(all_slices, slice_index):
    """ Returns the index of the lower slice of the given slice_index that is in the same lobe as the given index in the
        array of all slices. Returns the same slice if no lower slice in the lobe exist."""

    lobe_side = 'left'
    if 'right' in all_slices[slice_index]:
        lobe_side = 'right'

    i = slice_index + 1
    while i < len(all_slices) and lobe_side not in all_slices[i]:
        i += 1

    if i == len(all_slices):
        return slice_index
    else:
        return i


def find_good_slices(sample_path, subsample=False):
    """ Returns the list of slices dirs of the sample with the given path, sorted by their height.
    Slices with total area less than 15000 are filtered out (+ the noisy ones among a batch of small lobes!)
    If subsample is True, 160 slices would be subsampled evenly in the height ."""

    subdirs = listdir(sample_path)
    if len(subdirs) == 0:
        return np.asarray([])

    # Deciding IDs to keep

    subdirs = [sd for sd in subdirs if 'mask' in sd and 'inf' not in sd]
    if len(subdirs) == 0:
        return np.asarray([])
    #print('%d masks found' % len(subdirs))

    subdirs = np.asarray(subdirs, dtype=np.object)

    # sorting by height

    slice_heights = np.vectorize(lambda fd: float(path.basename(fd).split('_')[1]))(subdirs)
    sorted_args = np.argsort(slice_heights)
    subdirs = subdirs[sorted_args]

    # filtering by lung pixels

    num_format = regex.compile('\d*.?\d+')
    lung_pixels = np.vectorize(lambda fd: int(([x for x in path.basename(fd).split('_') if num_format.match(x)])[-1]))(subdirs)
    lids = np.arange(len(lung_pixels))
    small_lung_mask = (lung_pixels < 15000)

    # from middle to left choosing the last small lung that has another small lung in vicinity
    # from middle to right choosing the first small lung that has another small lung in vicinity
    # middle_index = int(len(lids) / 2)
    left_q_lim = int(len(lids) / 4)
    right_q_lim = int(3 * len(lids) / 4)
    # left_q_lim = int(len(lids) / 2)
    # right_q_lim = int(len(lids) / 2)

    has_small_lung_in_vicinity = (np.vectorize(
        lambda xi:
        xi == 0 or xi == len(lids) - 1 or
        np.any(small_lung_mask[max(0, xi - 2):xi]) or
        np.any(small_lung_mask[xi + 1: min(xi + 3, len(lids))])))

    smallness_confirmed = np.logical_and(
        small_lung_mask, has_small_lung_in_vicinity(lids))

    left_el = lids[np.logical_and(
        lids < left_q_lim, smallness_confirmed)]

    if len(left_el) == 0:
        li = -1
    else:
        li = left_el[-1]

    right_el = lids[np.logical_and(
        lids > right_q_lim, smallness_confirmed)]

    if len(right_el) == 0:
        ri = len(lids)
    else:
        ri = right_el[0]

    subdirs = subdirs[li + 1: ri]

    # The ids to be kept
    ids_to_be_kept = np.sort(np.vectorize(lambda fd: int(path.basename(fd).split('_')[0]))(subdirs))

    # Now reading left and right lobes

    # filter the ones not having left or right in their names
    subdirs = listdir(sample_path)
    subdirs = [sd for sd in subdirs if ('left' in sd or 'right' in sd)]
    subdirs = [sd for sd in subdirs if '_bd_' not in sd and '_inf_' not in sd and '_dist_' not in sd]

    subdirs = np.asarray(subdirs, dtype=np.object)

    if len(subdirs) == 0:
        return np.asarray([])

    # sorting by height

    slice_heights = np.vectorize(lambda fd: float(path.basename(fd).split('_')[1]))(subdirs)
    sorted_args = np.argsort(slice_heights)
    subdirs = subdirs[sorted_args]

    # filtering the ones their IDs have been filtered
    slices_ids = np.vectorize(lambda fd: float(path.basename(fd).split('_')[0]))(subdirs)
    keep_mask = (slices_ids ==
                                    ids_to_be_kept[np.minimum(
                                        len(ids_to_be_kept) - 1,
                                        np.searchsorted(ids_to_be_kept, slices_ids, side='left')
                                    )])
    subdirs = subdirs[keep_mask]
    #subdirs = subdirs[np.vectorize(lambda fd: float(path.basename(fd).split('_')[-1]))(subdirs) > 10000]

    return [sample_path + '/' + x for x in subdirs.tolist()]


def prepare_sample_per_slice(all_slices, slice_index, data_loader):
    """ Returns the required stack per slice, depending on whether the upper on lower slices should
    be attached or not. """

    conf = data_loader.conf
    train_phase = data_loader.train_phase

    if not conf['3_slice_samples']:
        im = read_single_img(all_slices[slice_index])
    else:  # no augmentation
        im = np.stack((
            read_single_img(all_slices[get_prev_slice_index(all_slices, slice_index)]),
            read_single_img(all_slices[slice_index]),
            read_single_img(all_slices[get_next_slice_index(all_slices, slice_index)]),
        ), axis=0)

    augmenter = conf['augmenter']

    if train_phase and augmenter is not None:
        return augmenter(im)
    else:
        return im

