from abc import abstractmethod
import torch
import numpy as np


class ContentLoader:
    """ A class for loading one content type needed for the run. """

    def __init__(self, conf, prefix_name, data_specification):
        """ Receives as input conf which is a dictionary containing configurations.
        This dictionary can also be used to pass fixed addresses for easing the usage!
        prefix_name is the str which all the variables that must be filled with this class have
        this prefix in their names so it would be clear that this class should fill it.
        data_specification is the string that specifies data and where it is! e.g. train, test, val"""

        self.conf = conf
        self.prefix_name = prefix_name

    @abstractmethod
    def get_samples_names(self):
        """ Returns a list containing names of all the samples of the content loader,
        each sample must owns a unique ID, and this function returns all this IDs.
        The order of the list must always be the same during one run.
        For example, this function can return an ID column of a table for TableLoader
         or the dir of images as ID for ImageLoader"""

    @abstractmethod
    def get_samples_labels(self):
        """ Returns list of labels of the whole samples.
        The order of the list must always be the same during one run."""

    @abstractmethod
    def get_samples_batch_effect_groups(self):
        """ Returns a dictionary from each class label to one list per class label.
        The list contains lists of indices of the samples related to one batch effect group, e.g.
        the ones captured in one hospital!"""

    def get_reordered_indices(self, old_samlpes_names, new_samples_names):
        """ Returns the indices orders that old_samples_names[indices] = new_samples_names"""

        ordered_samples = np.sort(old_samlpes_names)
        samples_indices_map = np.zeros((len(ordered_samples),), dtype=np.int)
        samples_indices_map[np.searchsorted(ordered_samples, old_samlpes_names, side='left')] = \
            np.arange(len(ordered_samples))

        reorder_inds = samples_indices_map[
            np.minimum(
                np.searchsorted(ordered_samples, new_samples_names, side='left'),
                len(ordered_samples) - 1)
        ]

        return reorder_inds

    @abstractmethod
    def reorder_samples(self, indices, new_names):
        """ Reorders the samples to match the given samples_names.
        (The order is given as input as a list of samples). So the indices would be the same in all content loaders."""

    @abstractmethod
    def get_views_indices(self):
        """ Views are separated samples belonging to one subject (one patient e.g.).
        This method returns a list of names containing names of the subjects and a list of lists
        containing indices of views for each subject.
        If there aren't different views, return list of sample names lists, [[sample name1], [sample name2], ...]"""

    @abstractmethod
    def get_placeholder_name_to_fill_function_dict(self):
        """ Returns a dictionary of the placeholders' names (the ones this content loader supports)
        to the functions used for filling them. The functions must receive as input data_loader,
        which is an object of class data_loader that contains information about the current batch
        (e.g. the indices of the samples, or if the sample has many elements the indices of the chosen
        elements) and return an array per placeholder name according to the receives batch information.
        IMPORTANT: Better to use a fixed prefix in the names of the placeholders to become clear which content loader
        they belong to! Some sort of having a mark :))!"""

    def fill_placeholders(self, data_loader, placeholders):
        """ Receives as input data_loader, which an object of class data_loader that contains information
        about the current batch (e.g. the indices of the samples, or if the sample has many elements
        the indices of the chosen elements), and placeholders which is a dictionary of the placeholders'
        names to a torch tensor for filling it to feed the model with. Fills placeholders based on the function dictionary
        received in get_placeholder_name_to_fill_function_dict."""

        placeholder_name_func_dict = dict()
        for k, v in self.get_placeholder_name_to_fill_function_dict().items():
            placeholder_name_func_dict[self.prefix_name + '_' + k] = v

        def fill_placeholder(placeholder, val):
            """ Fills the torch placeholder with the given numpy value,
            If shape mismatches, resizes the placeholder so the data would fit. """

            # Resize if the shape mismatches
            if list(placeholder.shape) != list(val.shape):
                placeholder.resize_(*tuple(val.shape))

            # feeding the value
            placeholder.copy_(torch.Tensor(val))

        # Filling all the placeholders in the received dictionary!
        for placeholder_name, placeholder_value in placeholders.items():
            if placeholder_name in placeholder_name_func_dict:
                placeholder_v = placeholder_name_func_dict[placeholder_name](data_loader)
                if placeholder_v is None:
                    print('Warning: None value for key %s' % placeholder_name)
                fill_placeholder(placeholder_value, placeholder_v)
            elif placeholder_name.startswith(self.prefix_name + '_'):
                print('Warning: placeholder %s having the same prefix of this class was not filled' % placeholder_name)

    def fix_multiple_views(self, views_dict, views_names):
        """ Fixes multi view names in the content loaders that do not have multi views,
        repeating their data so all indices can match in all samples. Views_names is a list of names of the views
        with the order of the given names in the function get_names"""
        pass

    def set_views_names(self, views_names):
        """ Sets views names in content loaders that do not have multiple views"""
        pass

    def validate(self):
        pass
