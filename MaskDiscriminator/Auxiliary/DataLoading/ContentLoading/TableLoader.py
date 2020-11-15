import pandas as pd
from Auxiliary.DataLoading.ContentLoading.ContentLoader import ContentLoader
import numpy as np


class TableLoader(ContentLoader):

    def __init__(self, conf, prefix_name, data_specification, table_directory):
        super(TableLoader, self).__init__(conf, prefix_name, data_specification)

        self.data = pd.read_csv(table_directory)
        self.features = self.data.columns
        self.features_dict = {v: k for k, v in enumerate(self.features)}
        non_ess = ['Hospitalization_Days', 'ICU_Days', 'Intubation']
        self.ess = [self.features_dict[k] for k in self.features_dict.keys() if k not in non_ess]
        self.non_ess = [self.features_dict[k] for k in self.features_dict.keys() if k in non_ess]
        self.ess.remove(0)
        del self.ess[-1]
        if 'Hospitalization_Days' in self.features and 'ICU_Days' in self.features and 'Intubation' in self.features:
            self.augment_data = True
        else:
            self.augment_data = False

        self.max_hospitalization_days = conf['max_day']
        self.max_icu_days = conf['max_day']
        self.specific_table = None
        self.data_specification = data_specification
        self.load_tables(data_specification)
        self.threshold = conf['augment_rate']
        self.new_batch = None
        self.original_indices = None
        self.new_batch_size = conf['batch_size']
        self.current_index = None
        # self.new_ids = None

    def load_tables(self, data_specification):

        data_specification = np.copy(data_specification)

        if data_specification in ['train', 'val', 'test']:
            data_specification = '../DataSeparation/%s/%s.txt' % \
                                 (self.conf['dataSeparation'], data_specification)
        ids = pd.read_csv(data_specification, sep=':', header=None).values

        views_base_name_ids = np.unique(np.vectorize(lambda x: x[:x.rfind('_')])(ids[:, 1]))

        self.specific_table = self.data
        table_base_names = np.vectorize(
            lambda x: x[:x.rfind('_')])(self.specific_table['ID'].values)

        valid_mask = (table_base_names ==
                      views_base_name_ids[np.minimum(
                          len(views_base_name_ids) - 1,
                          np.searchsorted(views_base_name_ids, table_base_names, side='left')
                      )])

        self.specific_table = self.specific_table[valid_mask]
        self.specific_table['index'] = np.arange(np.sum(valid_mask))
        self.specific_table.set_index('index', inplace=True)

    def get_samples_names(self):
        return self.specific_table['ID'].values

    def get_samples_labels(self):
        return self.specific_table['Label'].values

    def get_samples_batch_effect_groups(self):
        if 'BaseName' in list(self.specific_table.columns.values):
            return np.unique(self.specific_table['BaseName'].values)
        else:
            return self.specific_table['ID'].values

    def reorder_samples(self, indices, new_names):
        self.specific_table = self.specific_table.loc[indices]
        # self.specific_table.set_index(indices)
        # self.specific_table.sort_index()
        self.specific_table['ID'] = new_names
        self.specific_table['index'] = np.arange(len(indices))
        self.specific_table.set_index('index', inplace=True)

    def get_views_indices(self):
        return [[i] for i in range(self.specific_table.shape[0])]

    def get_placeholder_name_to_fill_function_dict(self):
        return {
            'table_features': self.get_table_features,
            'essential_features': self.get_essential_features,  # Sex and Age
            'non_essential_features': self.get_non_essential_features,  # Blood tests and etc except Sex and Age
            'life_status': self.get_life_status
        }

    def augment_add(self, specific_table):
        if specific_table.shape[0] == 0:
            raise Exception("Zero batch Erroe")

        self.current_index = 0
        self.new_batch = np.array([])
        self.original_indices = []
        # self.new_ids = []
        # if np.array(specific_table).shape[0] == 1:
        id_hos = self.features_dict['Hospitalization_Days'] - 1
        id_icu = self.features_dict['ICU_Days'] - 1
        id_intu = self.features_dict['Intubation'] - 1

        def augment_single_patient(row_index):
            if augmented[row_index]:
                return
            return_batch = np.repeat(specific_table[row_index, :][None, :].copy(), 4, axis=0)
            if specific_table[row_index, -1] == 1:  # Dead person
                if specific_table[row_index, id_intu] == 1:
                    return_batch[3, id_intu] = 0
                    return_batch[2, id_intu] = 0
                icu_days_patient = np.random.randint(0, specific_table[row_index, id_icu] + 1, 3)
                hos_days_patient = np.random.randint(icu_days_patient, specific_table[row_index, id_hos] + 1, 3)
                return_batch[3, id_icu] = icu_days_patient[2]
                return_batch[2, id_icu] = icu_days_patient[1]
                return_batch[1, id_icu] = icu_days_patient[0]
                return_batch[3, id_hos] = hos_days_patient[2]
                return_batch[2, id_hos] = hos_days_patient[1]
                return_batch[1, id_hos] = hos_days_patient[0]

            elif specific_table[row_index, -1] == 0:  # Alive person
                if specific_table[row_index, id_intu] == 0:
                    return_batch[3, id_intu] = 1
                    return_batch[2, id_intu] = 1
                icu_days_patient = np.random.randint(specific_table[row_index, id_icu] + 1, self.max_icu_days, 3)
                hos_days_patient = np.random.randint(
                    np.maximum(icu_days_patient, specific_table[row_index, id_hos] + 1), self.max_icu_days, 3)
                return_batch[3, id_icu] = icu_days_patient[2]
                return_batch[2, id_icu] = icu_days_patient[1]
                return_batch[1, id_icu] = icu_days_patient[0]
                return_batch[3, id_hos] = hos_days_patient[2]
                return_batch[2, id_hos] = hos_days_patient[1]
                return_batch[1, id_hos] = hos_days_patient[0]

            if self.new_batch.shape[0] == 0:
                self.new_batch = return_batch
            else:
                self.new_batch = np.append(self.new_batch, return_batch, axis=0)
            augmented[row_index] = True

        v_func = np.vectorize(augment_single_patient)
        augmented = [False for _ in range(specific_table.shape[0])]
        v_func(np.arange(specific_table.shape[0]))
        self.new_batch_size = self.new_batch.shape[0]
        return self.new_batch[:, :-1]

    def augment(self, specific_table_c):
        specific_table = specific_table_c.copy()
        threshold = self.threshold
        if self.augment_data:
            id_hos = self.features_dict['Hospitalization_Days'] - 1
            id_icu = self.features_dict['ICU_Days'] - 1
            id_intu = self.features_dict['Intubation'] - 1

            def augment_single_patient(row_index):
                if specific_table[row_index, -1] == 1:  # Dead person
                    if specific_table[row_index, id_intu] == 1:
                        if np.random.random(1)[0] > threshold:
                            specific_table[row_index, id_intu] = 0
                    if np.random.random(1)[0] > threshold:
                        specific_table[row_index, id_icu] = np.random.randint(specific_table[row_index, id_icu] + 1)
                    if np.random.random(1)[0] > threshold:
                        specific_table[row_index, id_hos] = np.random.randint(specific_table[row_index, id_hos] + 1)

                elif specific_table[row_index, -1] == 0:  # Alive person
                    if specific_table[row_index, id_intu] == 0:
                        if np.random.random(1)[0] > threshold:
                            specific_table[row_index, id_intu] = 1
                    if np.random.random(1)[0] > threshold:
                        specific_table[row_index, id_icu] = np.random.randint(specific_table[row_index, id_icu],
                                                                              self.max_icu_days)
                        if specific_table[row_index, id_hos] < specific_table[row_index, id_icu]:
                            specific_table[row_index, id_hos] = np.random.randint(specific_table[row_index, id_icu],
                                                                                  self.max_icu_days)
                    if np.random.random(1)[0] > threshold:
                        specific_table[row_index, id_hos] = np.random.randint(specific_table[row_index, id_hos],
                                                                              self.max_hospitalization_days)

            v_func = np.vectorize(augment_single_patient)
            v_func(np.arange(specific_table.shape[0]))
            return specific_table[:, :-1]

    def binary_augment(self, specific_table_c):
        specific_table = specific_table_c.copy()
        threshold = self.threshold
        if self.augment_data:
            id_hos = self.features_dict['Hospitalization_Days'] - 1
            id_icu = self.features_dict['ICU_Days'] - 1
            id_intu = self.features_dict['Intubation'] - 1

            def augment_single_patient(row_index):
                if specific_table[row_index, -1] == 1:  # Dead person
                    if specific_table[row_index, id_intu] == 1:
                        if np.random.random(1)[0] > threshold:
                            specific_table[row_index, id_intu] = 0
                    if specific_table[row_index, id_icu] == 1:
                        if np.random.random(1)[0] > threshold:
                            specific_table[row_index, id_icu] = 0
                    if specific_table[row_index, id_hos] == 1:
                        if np.random.random(1)[0] > threshold:
                            specific_table[row_index, id_hos] = 0

                elif specific_table[row_index, -1] == 0:  # Alive person
                    if specific_table[row_index, id_intu] == 0:
                        if np.random.random(1)[0] > threshold:
                            specific_table[row_index, id_intu] = 1
                    if specific_table[row_index, id_icu] == 0:
                        if np.random.random(1)[0] > threshold:
                            specific_table[row_index, id_icu] = 1
                            if specific_table[row_index, id_hos] < specific_table[row_index, id_icu]:
                                specific_table[row_index, id_hos] = specific_table[row_index, id_icu]
                    if specific_table[row_index, id_hos] == 0:
                        if np.random.random(1)[0] > threshold:
                            specific_table[row_index, id_hos] = 1

            v_func = np.vectorize(augment_single_patient)
            v_func(np.arange(specific_table.shape[0]))
            return specific_table[:, :-1]

    def binary_augment_add(self, specific_table):
        self.current_index = 0
        self.new_batch = np.array([])
        self.original_indices = []
        # self.new_ids = []
        # if np.array(specific_table).shape[0] == 1:
        id_hos = self.features_dict['Hospitalization_Days'] - 1
        id_icu = self.features_dict['ICU_Days'] - 1
        id_intu = self.features_dict['Intubation'] - 1

        def augment_single_patient(row_index):
            if augmented[row_index]:
                return
            return_batch = np.repeat(specific_table[row_index, :][None, :], 4, axis=0)
            current_index = 3
            self.original_indices.append(self.current_index)
            if specific_table[row_index, -1] == 1:  # Dead person
                if specific_table[row_index, id_intu] == 1:
                    new_row = specific_table[row_index, np.newaxis].copy()
                    new_row[0, id_intu] = 0
                    return_batch[current_index] = new_row
                    current_index -= 1
                    # self.new_ids.append(ids.iloc[row_index])
                if specific_table[row_index, id_icu] == 1:
                    new_row = specific_table[row_index, np.newaxis].copy()
                    new_row[0, id_icu] = 0
                    return_batch[current_index] = new_row
                    current_index -= 1
                    # self.new_ids.append(ids.iloc[row_index])

                if specific_table[row_index, id_hos] == 1:
                    new_row = specific_table[row_index, np.newaxis].copy()
                    new_row[0, id_hos] = 0
                    return_batch[current_index] = new_row
                    current_index -= 1
                    # self.new_ids.append(ids.iloc[row_index])

            elif specific_table[row_index, -1] == 0:  # Alive person
                if specific_table[row_index, id_intu] == 0:
                    new_row = specific_table[row_index, np.newaxis].copy()
                    new_row[0, id_intu] = 1
                    return_batch[current_index] = new_row
                    current_index -= 1
                    # self.new_ids.append(ids.iloc[row_index])

                if specific_table[row_index, id_icu] == 0:
                    new_row = specific_table[row_index, np.newaxis].copy()
                    new_row[0, id_icu] = 1
                    if new_row[0, id_hos] < new_row[0, id_icu]:
                        new_row[0, id_hos] = new_row[0, id_icu]
                    return_batch[current_index] = new_row
                    current_index -= 1
                    # self.new_ids.append(ids.iloc[row_index])
                if specific_table[row_index, id_hos] == 0:
                    new_row = specific_table[row_index, np.newaxis].copy()
                    new_row[0, id_hos] = 1
                    return_batch[current_index] = new_row
                    current_index -= 1
                    # self.new_ids.append(ids.iloc[row_index])
            if self.new_batch.shape[0] == 0:
                self.new_batch = return_batch
                self.current_index += return_batch.shape[0]
            else:
                self.current_index += return_batch.shape[0]
                self.new_batch = np.append(self.new_batch, return_batch, axis=0)
            augmented[row_index] = True

        v_func = np.vectorize(augment_single_patient)
        augmented = [False for _ in range(np.array(specific_table).shape[0])]
        v_func(np.arange(np.array(specific_table).shape[0]))
        self.new_batch_size = self.new_batch.shape[0]
        if not (np.sum(self.new_batch[0, :] != specific_table[0, :]) == 0):
            raise Exception("Wrong binary augment!")
        return self.new_batch[:, :-1]

    def get_table_features(self, batch_chooser):
        return self.specific_table.iloc[batch_chooser.get_current_batch_sample_indices(), 1:-1].values.astype(
            np.float64)

    # Sex and Age
    def get_essential_features(self, batch_chooser):
        return self.specific_table.iloc[batch_chooser.get_current_batch_sample_indices(), self.ess].values.astype(
            np.float64)

    # Blood tests and etc except Sex and Age
    def get_non_essential_features(self, batch_chooser):
        return self.specific_table.iloc[batch_chooser.get_current_batch_sample_indices(), self.non_ess].values.astype(
            np.float64)

    def get_life_status(self, batch_chooser):
        return self.specific_table.iloc[batch_chooser.get_current_batch_sample_indices(), -1].values.astype(
            np.float64)
