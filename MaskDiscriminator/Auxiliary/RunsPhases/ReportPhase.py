from Auxiliary.RunsPhases.Phase import Phase
from Auxiliary.ModelLoading.ModelLoading import load_best_model
from time import time
import traceback
from Auxiliary.DataLoading.DataLoader import DataLoader


class ReportPhase(Phase):

    def __init__(self, conf, model):
        super(ReportPhase, self).__init__('report', conf, model)

    def run_subclass(self):
        """ Runs the report generation for all the specified samples in samples_dir,
        for all the specified epochs."""

        # running evaluation for all samples on all specified epochs
        self.run_for_multiple_epochs(self.run_report_generation_for_all_samples)

    def run_report_generation_for_all_samples(self):
        """ Runs the report generation for all of them samples specified in samples_dir. """

        load_best_model(self.model, self.conf)

        for test_group_info in self.conf['samples_dir'].split(','):
            try:
                print('')
                print('>> Running evaluations for %s' % test_group_info)
                t1 = time()

                test_data_loader = DataLoader(self.conf, test_group_info, 'test')
                evaluator = self.instantiate_evaluator(test_data_loader)

                if 'final_version' in self.conf and self.conf['epoch'] is None:
                    report_dir = '%s/%s/%s.tsv' % (
                        self.conf['report_dir'],
                        self.conf['final_version'][self.conf['final_version'].rfind('/') + 1:self.conf['final_version'].rfind('.')],
                        test_group_info.replace(':', '_').replace('../', ''))
                elif self.conf['load_dir'] != self.conf['save_dir']:
                    report_dir = '%s/%s_e%s/%s.tsv' % (self.conf['report_dir'],
                                                       self.conf['load_dir'], self.conf['epoch'],
                                                       test_group_info.replace(':', '_').replace('../', ''))
                else:
                    report_dir = '%s/%s_%d_e%s/%s.tsv' % (self.conf['report_dir'],
                                                          self.conf['try_name'], self.conf['try_num'], self.conf['epoch'],
                                                          test_group_info.replace(':', '_').replace('../', ''))

                evaluator.create_sample_report(report_dir)

                print('Making report was done in %.2f secs.' % (time() - t1,))
            except Exception as e:
                print('Problem in %s' % test_group_info)
                track = traceback.format_exc()
                print(track)
