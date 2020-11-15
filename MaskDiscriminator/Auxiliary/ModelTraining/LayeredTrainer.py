import numpy as np
import torch
from time import time
from os import path, makedirs
from Auxiliary.ModelTraining.Trainer import Trainer
from Models.LayeredTrainedModel import LayeredTrainedModel


class LayeredTrainer(Trainer):

    def __init__(self, the_model, conf, load_prev=True):
        super(LayeredTrainer, self).__init__(the_model, conf, load_prev)

        if not isinstance(the_model, LayeredTrainedModel):
            raise Exception('The model should inherit from Model.LayeredTrainedModel')

        self.n_epochs_to_train_layers = None
        self.layer_id_to_train = 0
        self.current_layer_training_cnt = 0

        self.define_optimizer()

    # -------------- For being overwritten in subclasses if needed -------------- #

    def train_model_for_one_epoch(self, train_loader, epoch, iters_per_epoch):
        """ Receives as input, train data_loader which is a data_loader
        containing information about train samples and how to iterate over them, the epoch of training
        and the number of iterations per epoch. Trains the model for one epoch and saves
        the stats of the epoch."""

        t1 = time()

        s_time = time()

        # resetting train loader
        train_loader.reset()

        # Switching on the next layer if necessary
        if self.current_layer_training_cnt == self.n_epochs_to_train_layers[self.layer_id_to_train]:
            self.layer_id_to_train = (self.layer_id_to_train + 1) % len(self.n_epochs_to_train_layers)
            self.current_layer_training_cnt = 0

        if self.current_layer_training_cnt == 0:
            self.the_model.switch_layer_on(self.layer_id_to_train)

        # changing the model's status to training mode
        self.the_model.train()

        aggregated_eval_reqs = \
            np.zeros((self.train_evaluator.get_the_number_of_evaluation_requirements(),))

        epoch_loss = 0.0

        # initializing placeholders to do training with them
        placeholders_dict = self.model_runner.create_placeholders_for_model_input()

        t2 = time()
        if self.print_time:
            print('Pretrain time %.2f secs.' % (t2 - t1,))

        for i in range(int(round(np.ceil(iters_per_epoch), 0))):

            big_batch_loss = 0

            for opt in self.optimizer:
                opt.zero_grad()

            for b in range(self.conf['big_batch_size']):

                t1 = time()
                train_loader.prepare_next_batch()
                t2 = time()
                if self.print_time:
                    print('Preparing batch in %.2f secs.' % (t2 - t1,))

                t1 = time()
                # Filling the placeholders for running the model
                train_loader.fill_placeholders(placeholders_dict)
                t2 = time()
                if self.print_time:
                    print('Filling placeholders in %.2f secs.' % (t2 - t1,))

                # Running the model
                t1 = time()
                model_output = self.model_runner.run_model(placeholders_dict)
                t2 = time()
                if self.print_time:
                    print('Running model in %.2f secs.' % (t2 - t1,))

                batch_loss = model_output['loss']

                #print('\tE:%d, I:%d, L:%.4f' % (epoch, i, float(batch_loss)))
                #print('\t\t', train_loader.get_current_batch_sample_indices())
                #print('\t\t', train_loader.get_current_batch_elements_indices())

                if torch.isnan(batch_loss):
                    raise Exception('Nan loss at epoch %d iteration %d' % (epoch, i))

                # backpropagation
                if self.dev_name is not None:
                    batch_loss = batch_loss
                else:
                    batch_loss = (1.0 / batch_loss.shape[0]) * torch.ones(batch_loss.shape[0], device=self.the_device)

                t1 = time()
                batch_loss.backward()
                t2 = time()
                if self.print_time:
                    print('Backward in %.2f secs.' % (t2 - t1,))

                big_batch_loss += float(batch_loss) / self.conf['big_batch_size']

                # Detaching outputs
                new_model_output = dict()
                for k, v in model_output.items():
                    new_model_output[k] = v.detach()
                model_output = new_model_output

                # updating model's prediction
                batch_gt, batch_model_pred = self.train_evaluator.extract_ground_truth_and_predictions_for_batch(
                        model_output)
                if self.conf['track_preds_on_data']:
                    train_loader.update_model_preds_for_sample(batch_model_pred)

                # Aggregating evaluation requirements
                t1 = time()
                aggregated_eval_reqs = self.train_evaluator.aggregate_evaluation_requirements(
                    aggregated_eval_reqs,
                    self.train_evaluator.calculate_evaluation_requirements(
                        batch_gt, batch_model_pred))
                t2 = time()
                if self.print_time:
                    print('Aggregating evaluations in %.2f secs.' % (t2 - t1,))

            epoch_loss += (big_batch_loss / iters_per_epoch)

            # Clearing the optimizer
            t1 = time()
            self.optimizer[self.layer_id_to_train].step()
            self.optimizer[self.layer_id_to_train].zero_grad()
            t2 = time()
            if self.print_time:
                print('Zeroing grad in %.2f secs.' % (t2 - t1,))

        self.train_evals[epoch, :] = np.asarray([epoch_loss] +
                                                list(self.train_evaluator.calculate_evaluation_metrics(
                                                    aggregated_eval_reqs)))

        print('>>> Epoch %d training done in %.2f secs.' % (epoch, time() - s_time))
        self.current_layer_training_cnt += 1
        print('* %d epoch of training the layer %d *' %
              (self.current_layer_training_cnt, self.layer_id_to_train))

    def define_optimizer(self):
        """ Defines the list of optimizers, each for one part of training! """

        self.the_model.setup_values_for_training()
        opts_configs = self.the_model.get_optimizer_types_of_layers_of_parameters_for_training()
        opts_params = self.the_model.layers_of_parameters_list
        self.n_epochs_to_train_layers = self.the_model.get_n_epochs_to_train_each_layer()

        self.optimizer = [
            opts_configs[i][0](
                opts_params[i],
                lr=opts_configs[i][1],
                weight_decay=opts_configs[i][2])
            for i in range(len(opts_configs))]

    def load_required_parameters(self, checkpoint):
        """ Loads the state of all the optimizers from the checkpoint """
        for i in range(len(self.optimizer)):
            self.optimizer[i].load_state_dict(checkpoint['optimizer_state_dict_%d' % i])

        self.layer_id_to_train = checkpoint['layer_id_to_train']
        self.current_layer_training_cnt = checkpoint['current_layer_training_cnt']

        self.the_model.switch_layer_on(self.layer_id_to_train)

    def add_required_state_to_dictionary(self, save_dict):
        """ Adds the state of all the optimizers to the saving dictionary. + its required parameters!"""
        for i in range(len(self.optimizer)):
            save_dict['optimizer_state_dict_%d' % i] = self.optimizer[i].state_dict()

        save_dict['layer_id_to_train'] = self.layer_id_to_train
        save_dict['current_layer_training_cnt'] = self.current_layer_training_cnt

