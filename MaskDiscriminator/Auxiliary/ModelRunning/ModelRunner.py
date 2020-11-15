import torch
import inspect


class ModelRunner:
    """ Runner is an object used in different classes (e.g. trainers and evaluators)
    for running the models."""

    def __init__(self, model, conf):
        """ Receives as input the model to run and the configurations. """
        self.model = model
        self.conf = conf

    def create_placeholders_for_model_input(self):
        """ Creates placeholders for feeding the model. Returns a dictionary containing placeholders.
        The placeholders are created based on the names of the input variables of the model's forward method.
        The variables initialized as None are assumed to be labels used for training phase only,
        and won't be added in phases other than train."""

        model_forward_args = list(inspect.getfullargspec(self.model.forward).args)
        # skipping self arg
        model_forward_args = model_forward_args[1:]

        args_default_values = inspect.getfullargspec(self.model.forward).defaults
        if args_default_values is None:
            args_default_values = []
        else:
            args_default_values = list(args_default_values)

        # Checking which args have None as default
        args_default_values = ['None' for _ in
                               range(len(model_forward_args) - len(args_default_values))] + \
                              args_default_values

        train_args = [x is None for x in args_default_values]

        placeholders = dict()

        for i in range(len(model_forward_args)):
            # Filtering the args related to train phase in other phases
            if self.conf['phase'] == 'train' or not train_args[i]:
                # if a default value is given, giving it instead
                if args_default_values[i] != 'None' and args_default_values[i] is not None:
                    continue
                else:
                    # Defining a placeholder with shape 1, this will be reshaped when required in dataloader
                    placeholders[model_forward_args[i]] = \
                        torch.zeros(1, dtype=torch.float32, device=self.conf['device'], requires_grad=self.conf['require_grad_for_input'])

        return placeholders

    def run_model(self, placeholders_dict):
        """ Receives as input a dictionary containing the placeholders for the input of the model.
         The placeholders must be filled by the data_loader before calling this function.
         Runs the model based on the required inputs. Returns the outputs of the model. """

        return self.model(** placeholders_dict)

