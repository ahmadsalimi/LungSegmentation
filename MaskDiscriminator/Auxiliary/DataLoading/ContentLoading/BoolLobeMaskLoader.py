from os import listdir, path
import numpy as np
import re as regex
from Auxiliary.DataLoading.ContentLoading.ContentLoader import ContentLoader
from Auxiliary.Threading.WorkerCoordinating import WorkersCoordinator
import pandas as pd


class BoolLobeMaskLoader(ContentLoader):

    def __init__(self, conf, prefix_name, data_specification):
        super(BoolLobeMaskLoader, self).__init__(conf, prefix_name, data_specification)

    def get_samples_names(self):
        """ Returns a list containing names of all the samples of the content loader,
        each sample must owns a unique ID, and this function returns all this IDs.
        The order of the list must always be the same during one run.
        For example, this function can return an ID column of a table for TableLoader
         or the dir of images as ID for ImageLoader"""

    def get_samples_labels(self):
        """ Returns list of labels of the whole samples.
        The order of the list must always be the same during one run."""

    def get_samples_batch_effect_groups(self):
        """ Returns a dictionary from each class label to one list per class label.
        The list contains lists of indices of the samples related to one batch effect group, e.g.
        the ones captured in one hospital!"""

    def reorder_samples(self, indices, new_names):
        """ Reorders the samples to match the given samples_names.
        (The order is given as input as a list of samples). So the indices would be the same in all content loaders."""

    def get_views_indices(self):
        """ Views are separated samples belonging to one subject (one patient e.g.).
        This method returns a list of names containing names of the subjects and a list of lists
        containing indices of views for each subject.
        If there aren't different views, return list of sample names lists, [[sample name1], [sample name2], ...]"""

    def get_placeholder_name_to_fill_function_dict(self):
        """ Returns a dictionary of the placeholders' names (the ones this content loader supports)
        to the functions used for filling them. The functions must receive as input data_loader,
        which is an object of class data_loader that contains information about the current batch
        (e.g. the indices of the samples, or if the sample has many elements the indices of the chosen
        elements) and return an array per placeholder name according to the receives batch information.
        IMPORTANT: Better to use a fixed prefix in the names of the placeholders to become clear which content loader
        they belong to! Some sort of having a mark :))!"""

