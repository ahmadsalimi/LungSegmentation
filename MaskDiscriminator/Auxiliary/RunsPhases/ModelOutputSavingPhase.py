from Auxiliary.RunsPhases.Phase import Phase
from Auxiliary.ModelLoading.ModelLoading import load_best_model
from time import time
import traceback
from Auxiliary.DataLoading.DataLoader import DataLoader


class ModelOutputSavingPhase(Phase):

    def __init__(self, conf, model):
        super(ModelOutputSavingPhase, self).__init__('eval', conf, model)

    def run_subclass(self):
        """ Runs the evaluation for all the specified samples in samples_dir, for all the specified epochs."""
        # running evaluation for all samples on all specified epochs
        self.run_for_multiple_epochs(self.run_eval_for_all_samples)

    def run_eval_for_all_samples(self):
        """ Runs the evaluation for all of them samples specified in samples_dir. """

        load_best_model(self.model, self.conf)

        for test_group_info in self.conf['samples_dir'].split(','):
            try:
                print('')
                print('>> Running saving model output for %s' % test_group_info)

                t1 = time()

                test_data_loader = DataLoader(self.conf, test_group_info, 'test')
                evaluator = self.instantiate_evaluator(test_data_loader)
                evaluator.save_model_outputs()

                print('Saving the output of the model was done in %.2f secs.' % (time() - t1,))
            except Exception as e:
                print('Problem in %s' % test_group_info)
                track = traceback.format_exc()
                print(track)



