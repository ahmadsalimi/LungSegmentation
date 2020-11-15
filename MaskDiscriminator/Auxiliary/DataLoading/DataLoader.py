import numpy as np


class DataLoader:
    """ A class for loading the whole data needed for the run! """

    def __init__(self, conf, sample_specification, run_type):
        """ Conf is a dictionary containing configurations,
        sample specification is a string specifying the samples e.g. address of the CTs
        run type is one of the strings train/val/test specifying the mode that the data_loader
        will be used."""

        self.conf = conf
        self.sample_specification = sample_specification

        self.content_loaders = []
        for cl_maker, cl_vals in self.conf['content_loaders']:
            inp_dict = cl_vals.copy()
            inp_dict.update({'conf': self.conf, 'data_specification': sample_specification})
            self.content_loaders.append(cl_maker(** inp_dict))

        self.samples_names = None
        self.samples_labels = None
        self.samples_batch_effect_groups = None
        self.class_samples_indices = None

        self.coordinate_samples()

        # setting stuff
        if self.samples_names is None:
            self.samples_names = self.content_loaders[0].get_samples_names()
        self.samples_labels = self.content_loaders[0].get_samples_labels()
        self.class_samples_indices = dict()
        print('%d samples' % len(self.samples_names))
        for i in range(len(self.samples_names)):
            if self.samples_labels[i] not in self.class_samples_indices:
                self.class_samples_indices[self.samples_labels[i]] = []
            self.class_samples_indices[self.samples_labels[i]].append(i)

        for c in self.class_samples_indices.keys():
            self.class_samples_indices[c] = np.asarray(self.class_samples_indices[c])

        # Creating batch chooser

        sample_chooser_maker, element_chooser_maker = self.conf[run_type + '_sampler']
        element_chooser = None
        if element_chooser_maker is not None:
            if type(element_chooser_maker) != list:
                element_chooser = element_chooser_maker(self)
            else:
                element_chooser = [(p, emaker(self)) for p, emaker in element_chooser_maker]

        self.batch_chooser = sample_chooser_maker(self.conf, self, element_chooser)

        self.model_preds_for_samples = np.abs(1 - self.samples_labels)  # for the first time
        self.model_preds_for_smaples_elements = None

    def validate(self):
        for cl in self.content_loaders:
            cl.validate()

    def coordinate_samples(self):

        if len(self.content_loaders) == 1:
            self.samples_names = self.content_loaders[0].get_samples_names()
            return

        all_cls_samples_names = [cl.get_samples_names() for cl in self.content_loaders]

        # finding base names for each of them!
        base_name_finder = np.vectorize(lambda x: x[:x.rfind('_')])
        all_cls_base_names = [base_name_finder(asn) for asn in all_cls_samples_names]

        # making a dict from each base name to the index of the samples
        name_index_dict = [dict() for _ in range(len(all_cls_base_names))]
        for i in range(len(all_cls_base_names)):
            for j in range(len(all_cls_base_names[i])):
                bsn = all_cls_base_names[i][j]
                if bsn not in name_index_dict[i]:
                    name_index_dict[i][bsn] = []
                name_index_dict[i][bsn].append(j)

        # using intersection of all base names as the final samples to use
        unified_base_names = np.unique(all_cls_base_names[0])
        for i in range(1, len(all_cls_base_names)):
            unified_base_names = np.intersect1d(
                unified_base_names,
                np.unique(all_cls_base_names[i]))

        # now extending all possibilities for all of the base names!
        new_samples_inds_in_all_cls = [[] for _ in range(len(self.content_loaders))]
        new_samples_multiplied_names = []

        for ubase_name in unified_base_names:
            inds_in_cls = [(x,) for x in name_index_dict[0][ubase_name]]

            for i in range(1, len(name_index_dict)):
                new_inds_in_cls = [x + (y,) for x in inds_in_cls for y in name_index_dict[i][ubase_name]]
                inds_in_cls = new_inds_in_cls

            new_samples_multiplied_names += [ubase_name + ''.join([
                all_cls_samples_names[j][inds_in_cls[i][j]].replace(ubase_name, '')
                for j in range(len(all_cls_samples_names))
            ]) for i in range(len(inds_in_cls))]

            for i in range(len(new_samples_inds_in_all_cls)):
                new_samples_inds_in_all_cls[i] += [x[i] for x in inds_in_cls]

        for i in range(len(new_samples_inds_in_all_cls)):
            new_samples_inds_in_all_cls[i] = np.asarray(new_samples_inds_in_all_cls[i])

        self.samples_names = new_samples_multiplied_names

        for cli in range(len(new_samples_inds_in_all_cls)):
            self.content_loaders[cli].reorder_samples(
                new_samples_inds_in_all_cls[cli],
                new_samples_multiplied_names)

    def get_samples_names(self):
        """ Returns the names of the samples based on the first content loader.
        IMPORTANT: These all must be the same for all of the content loaders."""
        return self.samples_names

    def get_samples_labels(self):
        """ Returns the labels of the samples in a numpy array. """
        return self.samples_labels

    def get_number_of_samples(self):
        """ Returns the number of samples loaded. """
        return len(self.samples_names)

    def get_class_sample_indices(self):
        """ Returns a dictionary, containing lists of samples indices belonging to each class label."""
        return self.class_samples_indices

    def reset(self):
        """ Resets the state of the dataloader, here the batch_chooser."""
        self.batch_chooser.reset()

    def prepare_next_batch(self):
        """ Prepares the next batch and saves the information about it."""
        self.batch_chooser.prepare_next_batch()

    def finished_iteration(self):
        """ Returns True if batch chooser has done iterating, e.g. has iterated over all the samples. """
        return self.batch_chooser.finished_iteration()

    def fill_placeholders(self, placeholders_dict):
        """ Receives as input a dictionary of placeholders and fills them using all the content loaders."""
        for cl in self.content_loaders:
            cl.fill_placeholders(self, placeholders_dict)

    def get_current_batch_sample_indices(self):
        """ Returns a list of indices of the samples chosen for the current batch. """
        return self.batch_chooser.get_current_batch_sample_indices()

    def get_current_batch_elements_indices(self):
        """ Returns a list of lists, one list per sample containing lists of the chosen elements
         of the samples chosen for the current batch. """
        return self.batch_chooser.get_current_batch_elements_indices()

    def update_model_preds_for_sample(self, preds_for_batch):
        self.model_preds_for_samples[self.get_current_batch_sample_indices()] = preds_for_batch

    def initiate_model_preds_for_samples_elements(self, init_vals):
        self.model_preds_for_smaples_elements = init_vals
        for i in range(len(self.model_preds_for_samples)):
            if len(self.model_preds_for_smaples_elements[i]) > 0:
                self.model_preds_for_samples[i] = np.amax(self.model_preds_for_smaples_elements[i])
            else:
                self.model_preds_for_samples[i] = 0

    def update_model_preds_for_samples_elements(self, preds_for_batch):
        batch_sample_inds = self.get_current_batch_sample_indices()
        batch_elements_inds = self.get_current_batch_elements_indices()

        for i in range(len(batch_sample_inds)):
            ok_mask = batch_elements_inds[i, :] > -1
            self.model_preds_for_smaples_elements[
                batch_sample_inds[i]][(batch_elements_inds[i, :])[ok_mask]] = \
                (preds_for_batch[i, :])[ok_mask]

