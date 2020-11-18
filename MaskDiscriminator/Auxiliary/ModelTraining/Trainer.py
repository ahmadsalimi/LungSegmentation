import numpy as np
import torch
from time import time
from os import path, makedirs


class Trainer:

    def __init__(self, the_model, conf, load_prev=True):

        self.print_time = False

        self.conf = conf
        self.the_model = the_model
        
        self.load_prev = load_prev

        self.train_evals = None
        self.val_evals = None

        self.debug_mode = conf['debug_mode']

        self.batch_size = self.conf['batch_size']
        self.elements_per_batch = self.conf['elements_per_batch']
        self.inp_size = self.conf['inp_size']
        self.dev_name = self.conf['dev_name']
        self.the_device = self.conf['device']
        self.model_ptr = self.the_model

        self.train_evaluator = None
        self.val_evaluator = None

        self.model_runner = self.conf['model_runner'](the_model, conf)

        self.optimizer = None
        self.define_optimizer()

    def load_last_saved_training_stats(self, training_epochs):
        """ Loads the training stats from the last saved epoch to continue training on it. """

        save_dir = self.conf['save_dir']

        if not path.exists(save_dir):
            makedirs(save_dir)

        best_val_epoch = 0
        start_epoch = 0

        if path.exists(save_dir + '/GeneralInfo'):

            checkpoint = torch.load(save_dir + '/GeneralInfo')
            start_epoch = checkpoint['epoch'] + 1

            if 'epoch' in self.conf and self.conf['epoch'] is not None:
                start_epoch = int(self.conf['epoch']) + 1

            self.load_required_parameters(checkpoint)

            prev_train_evals = np.load(save_dir + '/train_evals.npy')
            prev_val_evals = np.load(save_dir + '/val_evals.npy')

            if training_epochs <= len(prev_train_evals):
                self.train_evals = prev_train_evals[:training_epochs]
                self.val_evals = prev_val_evals[:training_epochs]
            else:
                self.train_evals[: len(prev_train_evals)] = prev_train_evals
                self.val_evals[: len(prev_val_evals)] = prev_val_evals

            best_val_epoch = np.argmin(self.val_evals[:, 0])

            checkpoint = torch.load('%s/%d' % (save_dir, start_epoch - 1,))
            self.model_ptr.load_state_dict(checkpoint['model_state_dict'])

            for i in range(start_epoch):
                print('*** Epoch %d, train loss: %.4e, %s ***' % (i, self.train_evals[i, 0], ', '.join([
                    '%s: %.2f' % (eval_title, eval_value) for (eval_title, eval_value) in
                    zip(self.train_evaluator.get_headers_of_evaluation_metrics(), self.train_evals[i, 1:])])))
                print('*** Epoch %d, val loss: %.4e, %s ***' % (i, self.val_evals[i, 0], ', '.join([
                    '%s: %.2f' % (eval_title, eval_value) for (eval_title, eval_value) in
                    zip(self.train_evaluator.get_headers_of_evaluation_metrics(), self.val_evals[i, 1:])])))

        return best_val_epoch, start_epoch

    def set_iters_per_epoch(self, train_sampler):
        """ Sets the number of iterations per epoch, if defined in the configurations to the defined value,
        otherwise to some value to go over the whole samples at least once. """

        self.batch_size = self.conf['batch_size']
        self.elements_per_batch = self.conf['elements_per_batch']

        if self.conf['iters_per_epoch'] is None:
            iters_per_epoch = int(np.ceil(2.0 * train_sampler.get_max_group_samples_num() / self.batch_size)) * \
                              int(np.ceil(1.0 * train_sampler.get_max_slices_num() / self.elements_per_batch))
        else:
            iters_per_epoch = self.conf['iters_per_epoch']

        return iters_per_epoch

    def save_training_stats(self, epoch, best_val_epoch):
        """ Saves the training stats related to the last epoch of training and the
        trained model at the end pf the epoch, updates best val epoch.
        Best val epoch used to be used for keeping that epoch only and
        eliminating the model saved in the other epochs but the feature is commented now
        as we don't use exact evaluating for validation data. """

        save_dir = self.conf['save_dir']

        save_dict = {'epoch': epoch}
        self.add_required_state_to_dictionary(save_dict)
        torch.save(save_dict, save_dir + '/GeneralInfo')

        np.save(save_dir + '/train_evals', self.train_evals)
        np.save(save_dir + '/val_evals', self.val_evals)

        epoch_save_dir = '%s/%d' % (save_dir, epoch)

        # saving the model
        torch.save({
            'model_state_dict': self.model_ptr.state_dict(),
        }, epoch_save_dir)

        # updating best val epoch
        if self.val_evals[epoch, 0] < self.val_evals[best_val_epoch, 0]:
            # removing prev best val's model
            # if path.exists('%s/%d' % (save_dir, best_val_epoch)):
            #    remove('%s/%d' % (save_dir, best_val_epoch))
            print('@ Changing best val epoch from %d, %.4e to %.4e' % (
                best_val_epoch, self.val_evals[best_val_epoch, 0], self.val_evals[epoch, 0]))
            best_val_epoch = epoch

        return best_val_epoch

    def save_log_file(self, epoch):
        """ Appends the evaluation metrics related to the given epoch to the end of the file. """

        log_dir = self.conf['save_dir'] + '/log.csv'

        # Writing the headers if the file does not exist
        if not path.exists(log_dir):
            f = open(log_dir, 'w')
            f.write(','.join(
                ['epoch', 'train_loss'] +
                ['train_%s' % x for x in self.train_evaluator.get_headers_of_evaluation_metrics()] +
                ['val_loss'] +
                ['val_%s' % x for x in self.train_evaluator.get_headers_of_evaluation_metrics()]) + '\n')
        else:
            f = open(log_dir, 'a')

        # appending the information of the current epoch
        # Changing pandas to some cave age code for HPC!
        f.write(','.join(
            [str(epoch)] +
            ['%.4f' % x for x in list(self.train_evals[epoch, :])] +
            ['%.4f' % x for x in list(self.val_evals[epoch, :])]
        ) + '\n')

        f.close()

    def set_up_training_parameters(self, train_sampler):
        """ Sets up the required parameters for training before it starts,
        loads the training from where it has stopped to continue it. Returns
        the number of iterations per epoch, the total number of training epochs,
        the best epoch till now based on validation loss and the start epoch
        (the last one training has been stopped there.)"""

        iters_per_epoch = self.set_iters_per_epoch(train_sampler)

        if self.debug_mode:
            iters_per_epoch = 1
            training_epochs = 1
        else:
            training_epochs = self.conf['max_epochs']

        if self.dev_name is not None and self.dev_name != 'cpu':
            self.the_model.cuda(int(self.dev_name.split(':')[1]))
        elif self.dev_name is None:
            self.the_model = torch.nn.DataParallel(self.the_model)
            self.the_model.cuda()

        n_eval_metrics = 1 + len(self.train_evaluator.get_headers_of_evaluation_metrics())
        self.train_evals = np.zeros((training_epochs, n_eval_metrics))
        self.val_evals = np.zeros((training_epochs, n_eval_metrics))

        # Loading previous state

        if self.load_prev:
            best_val_epoch, start_epoch = self.load_last_saved_training_stats(training_epochs)
        else:
            start_epoch = 0
            best_val_epoch = 0

        if start_epoch == 0:
            self.model_ptr.init_weights_from_other_model()

        return iters_per_epoch, training_epochs, best_val_epoch, start_epoch

    def after_train(self):
        """ Does the final stuff, frees the memory allocated. """

        del self.optimizer

    def train(self, train_loader, val_loader):
        """ Receives as input train_loader and val_loader which are data_loaders
        containing information about training samples and validation samples and how to
        iterate over them. Does the training on the received model in initializer and saves the
        model trained in each epoch."""

        # Making evaluators
        self.train_evaluator = self.conf['evaluator'](self.the_model, self.conf, train_loader)
        self.val_evaluator = self.conf['evaluator'](self.the_model, self.conf, val_loader)

        # setting the required parameters for training
        iters_per_epoch, training_epochs, best_val_epoch, start_epoch = \
            self.set_up_training_parameters(train_loader)

        print('For training: Big batch size is: ', self.conf['big_batch_size'])

        for epoch in range(start_epoch, training_epochs):
            s_time = time()

            # training the model
            self.train_model_for_one_epoch(train_loader, epoch, iters_per_epoch)

            # evaluating validation data for the trained model
            t1 = time()
            val_stats, val_loss = self.val_evaluator.evaluate(None)
            self.val_evals[epoch, :] = np.asarray([val_loss] + val_stats.tolist())
            t2 = time()
            if self.print_time:
                print('Val evaluations in %.2f secs.' % (t2 - t1,))

            t1 = time()

            # printing the stats
            print('*** Epoch %d, train loss: %.4e, %s ***' % (epoch, self.train_evals[epoch, 0], ', '.join([
                '%s: %.2f' % (eval_title, eval_value) for (eval_title, eval_value) in
                zip(self.train_evaluator.get_headers_of_evaluation_metrics(), self.train_evals[epoch, 1:])])))

            print('*** Epoch %d, val loss: %.4e, %s ***' % (epoch, self.val_evals[epoch, 0], ', '.join([
                '%s: %.2f' % (eval_title, eval_value) for (eval_title, eval_value) in
                zip(self.train_evaluator.get_headers_of_evaluation_metrics(), self.val_evals[epoch, 1:])])))

            print('')

            # Updating the log file
            self.save_log_file(epoch)

            # If debug mode, one iteration is enough
            if self.debug_mode and epoch > 0:
                break

            # Saving epoch
            best_val_epoch = self.save_training_stats(epoch, best_val_epoch)

            # removing prev epoch if it is not removed and is not the best
            # if path.exists('%s/%d' % (save_dir, epoch - 1)) and (epoch - 1 != best_val_epoch):
            #    remove('%s/%d' % (save_dir, epoch - 1))

            print('*** Epoch %d ended in %.2f secs ***' % (epoch, time() - s_time))
            t2 = time()
            if self.print_time:
                print('Saving stuff in %.2f secs.' % (t2 - t1,))
        self.after_train()

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

            self.optimizer.zero_grad()

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

                batch_loss /= self.conf['big_batch_size']

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
            self.optimizer.step()
            self.optimizer.zero_grad()
            t2 = time()
            if self.print_time:
                print('Zeroing grad in %.2f secs.' % (t2 - t1,))
            
            iter_evals = self.train_evaluator.calculate_evaluation_metrics(aggregated_eval_reqs)
            print('*** Iter %d, loss: %.4e, %s ***' % (i, epoch_loss * iters_per_epoch / (i + 1), ', '.join([
                    '%s: %.2f' % (eval_title, eval_value) for (eval_title, eval_value) in
                    zip(self.train_evaluator.get_headers_of_evaluation_metrics(), iter_evals)])), flush=True)

        self.train_evals[epoch, :] = np.asarray([epoch_loss] +
                                                list(self.train_evaluator.calculate_evaluation_metrics(
                                                    aggregated_eval_reqs)))

        print('>>> Epoch %d training done in %.2f secs.' % (epoch, time() - s_time))

    def define_optimizer(self):
        """ Defines the required optimizers for the training.
        Default is one optimizer based on the predefined arguments
        If no optimizer is specified the default is Adam(1e-4,decay=1e-6)"""
        
        default_init_lr = 1e-4
        default_decay = 1e-6
        
        if 'init_lr' not in self.conf:
            init_lr = default_init_lr
        else:
            init_lr = self.conf['init_lr']
            
        if 'lr_decay' not in self.conf:
            lr_decay = default_decay
        else:
            lr_decay = self.conf['lr_decay']
            
        # instantiating the optimizer
        if 'optimizer' in self.conf:
            self.optimizer = self.conf['optimizer'](filter(
                lambda p: p.requires_grad, self.model_ptr.parameters()),
                lr=init_lr, weight_decay=lr_decay)
        else:
            self.optimizer = torch.optim.Adam(filter(
                lambda p: p.requires_grad, self.model_ptr.parameters()),
                lr=init_lr, weight_decay=lr_decay)

    def load_required_parameters(self, checkpoint):
        """ Checkpoint is a dictionary keeping values of important parameters of training.
        It has been loaded from the specified location and this function is to load the
        values related to the optimizer and all other subclass dependent required things
        from that dictionary. """
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def add_required_state_to_dictionary(self, save_dict):
        """ This function adds the important values, e.g. state dictionary of all the used
        optimizers and all other subclass dependent required things
        to the dictionary passed to it so they will be saved in the process of
        saving the training parameters."""
        save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
