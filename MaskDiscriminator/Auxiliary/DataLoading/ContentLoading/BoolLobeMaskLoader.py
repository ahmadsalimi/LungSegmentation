from os import listdir, path
import numpy as np
import re as regex
from Auxiliary.DataLoading.BatchChoosing.BatchChooser import BatchChooser
from Auxiliary.DataLoading.ContentLoading.ContentLoader import ContentLoader
from Auxiliary.Threading.WorkerCoordinating import WorkersCoordinator
import pandas as pd
import torch
import torch.nn.functional as F


class BoolLobeMaskLoader(ContentLoader):

    def __init__(self, conf, prefix_name, data_specification):
        super(BoolLobeMaskLoader, self).__init__(
            conf, prefix_name, data_specification)
        filename = conf['dataSeparation']
        self.samples: pd.DataFrame = pd.read_csv(filename)
        self.samples: pd.DataFrame = self.samples[self.samples.Group == data_specification]
        self.loader_workers = WorkersCoordinator(4)

    def get_samples_names(self):
        """ Returns a list containing names of all the samples of the content loader,
        each sample must owns a unique ID, and this function returns all this IDs.
        The order of the list must always be the same during one run.
        For example, this function can return an ID column of a table for TableLoader
         or the dir of images as ID for ImageLoader"""
        return self.samples.Path.values

    def get_samples_labels(self):
        """ Returns list of labels of the whole samples.
        The order of the list must always be the same during one run."""
        return self.samples.Label.values

    def get_samples_batch_effect_groups(self):
        """ Returns a dictionary from each class label to one list per class label.
        The list contains lists of indices of the samples related to one batch effect group, e.g.
        the ones captured in one hospital!"""
        pass

    def reorder_samples(self, indices, new_names):
        """ Reorders the samples to match the given samples_names.
        (The order is given as input as a list of samples). So the indices would be the same in all content loaders."""
        self.samples = self.samples.loc[indices]
        self.samples['index'] = np.arange(len(indices))
        self.samples.set_index('index', inplace=True)

    def get_views_indices(self):
        """ Views are separated samples belonging to one subject (one patient e.g.).
        This method returns a list of names containing names of the subjects and a list of lists
        containing indices of views for each subject.
        If there aren't different views, return list of sample names lists, [[sample name1], [sample name2], ...]"""
        return self.get_samples_names(), np.arange(len(self.samples)).reshape((len(self.samples), 1))

    def get_placeholder_name_to_fill_function_dict(self):
        """ Returns a dictionary of the placeholders' names (the ones this content loader supports)
        to the functions used for filling them. The functions must receive as input data_loader,
        which is an object of class data_loader that contains information about the current batch
        (e.g. the indices of the samples, or if the sample has many elements the indices of the chosen
        elements) and return an array per placeholder name according to the receives batch information.
        IMPORTANT: Better to use a fixed prefix in the names of the placeholders to become clear which content loader
        they belong to! Some sort of having a mark :))!"""
        return {
            'sample': self.read_batch,
            'label': self.get_batch_label
        }

    def read_batch(self, batch_chooser: BatchChooser) -> np.ndarray:
        sample_inds: np.ndarray = batch_chooser.get_current_batch_sample_indices()
        element_inds: np.ndarray = batch_chooser.get_current_batch_elements_indices()

        def read_one_sample(sample_index: int, element_inds: np.ndarray) -> np.ndarray:
            filename = self.samples.Path.values[sample_index]
            sample = torch.tensor(np.load(filename))
            return F.interpolate(sample.float(), 256, mode='bilinear', align_corners=False).numpy()

        return np.stack(tuple(self.loader_workers.run_func(read_one_sample, zip(sample_inds, element_inds))), axis=0)

    def get_batch_label(self, batch_chooser: BatchChooser) -> np.ndarray:
        sample_inds: np.ndarray = batch_chooser.get_current_batch_sample_indices()
        return self.samples.Label.values[sample_inds]