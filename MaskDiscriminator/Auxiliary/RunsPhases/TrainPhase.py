from Auxiliary.RunsPhases.Phase import Phase
from Auxiliary.Configurations.Configs import set_random_seeds
from Auxiliary.ModelLoading.ModelLoading import load_best_model
from Auxiliary.DataLoading.DataLoader import DataLoader


class TrainPhase(Phase):

    def __init__(self, conf, model):
        super(TrainPhase, self).__init__('train', conf, model)

    def run_subclass(self):
        """ Loads the model from load_dir if load_dir is different from save_dir and trains the model. """

        trainer = self.instantiate_trainer()

        # Fixing random seeds
        set_random_seeds(self.conf['m_seed'])

        train_data_loader = DataLoader(self.conf, 'train', 'train')
        val_data_loader = DataLoader(self.conf, 'val', 'val')

        if self.conf['load_dir'] != self.conf['save_dir']:
            load_best_model(self.model, self.conf)

        trainer.train(train_data_loader, val_data_loader)
