from Auxiliary.RunsPhases.Phase import Phase
from Auxiliary.ModelLoading.ModelLoading import load_best_model
from time import time
import traceback
from Auxiliary.DataLoading.DataLoader import DataLoader


class SampleSliceReportPhase(Phase):

    def __init__(self, conf, model):
        super(SampleSliceReportPhase, self).__init__('savesliceprobs', conf, model)

    def run_subclass(self):
        """ Saves the corona probabilities assigned by the model for all slices in the directory specified,
         for all the specified samples."""

        load_best_model(self.model, self.conf)

        for test_group_info in self.conf['samples_dir'].split(','):
            try:

                print('')
                print('>> Saving the probability of slices for %s' % test_group_info)
                t1 = time()

                test_data_loader = DataLoader(self.conf, test_group_info, 'test')
                evaluator = self.instantiate_evaluator(test_data_loader)

                evaluator.create_report_for_sample_and_its_elements(self.conf['report_dir'])

                print('Saving the probability of slices was done in %.2f secs.' % (time() - t1,))
            except Exception as e:
                print('Problem in %s' % test_group_info)
                track = traceback.format_exc()
                print(track)
