from Models.Model import Model
from abc import abstractmethod
from torch.optim import Adam


class LayeredTrainedModel(Model):

    def __init__(self):
        super(LayeredTrainedModel, self).__init__()

        self.layers_of_parameters_list = None

    def setup_values_for_training(self):
        """ Sets up the required stuff in the model for the start of the work! """
        self.layers_of_parameters_list = self.get_layers_of_parameters_for_training()

        for p in self.parameters():
            p.requires_grad = False

    @abstractmethod
    def get_keywords_for_layers_of_parameters_for_training(self):
        """ Returns a list of list of strings, one for each layer of training.
        All the parameters containing the substring of one layer would be
        added to that layer of training (unfreezed in that training round)"""
        pass

    def get_layers_of_parameters_for_training(self):
        """ Expands the list of keywords and adds all the parameters matching them. """
        layers_keywords = self.get_keywords_for_layers_of_parameters_for_training()
        layers_params = [[] for _ in range(len(layers_keywords))]

        for name, param in self.named_parameters():
            for i in range(len(layers_keywords)):
                for keywd in layers_keywords[i]:
                    if keywd in name:
                        layers_params[i].append(param)
                        break

        return layers_params

    def get_optimizer_types_of_layers_of_parameters_for_training(self):
        """ Returns a list of tuples, the constructor of optimizer, initial learning rate and
        learning rate decay(0 if no decay). Default is all Adams with lr=1e-4, decay of 1e-6"""
        return [(Adam, 1e-4, 1e-6)
                for _ in range(len(self.get_keywords_for_layers_of_parameters_for_training()))]

    def get_n_epochs_to_train_each_layer(self):
        """ Returns a list of the number of epochs for training each layer, default is one for all """
        return [1 for _ in range(len(self.get_keywords_for_layers_of_parameters_for_training()))]

    def switch_layer_on(self, layer_id):
        """ Switches the previous layer off and unfreezes all the parameters of the given layer. """

        prev_layer = layer_id - 1
        if prev_layer == -1:
            prev_layer = len(self.layers_of_parameters_list) - 1

        for p in self.layers_of_parameters_list[prev_layer]:
            p.requires_grad = False

        for p in self.layers_of_parameters_list[layer_id]:
            p.requires_grad = True
