import torch


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def init_weights_from_other_model(self, pretrained_model_dir=None):
        """ Initializes the weights from another model with the given address,
        Default is to set all the parameters with the same name and shape,
        but can be rewritten in subclasses.
        The model to load is received via the function get_other_model_to_load_from.
        The default value is the same model!"""

        if pretrained_model_dir is None:
            print('The model was not preinitialized.')
            return

        other_model = self.get_other_model_to_load_from()

        if other_model == self:
            self.load_state_dict(torch.load(pretrained_model_dir))
            print('The whole state dictionary loaded successfully.')
            return

        cnt = 0

        other_model.load_state_dict(torch.load(pretrained_model_dir))
        pm = other_model.state_dict()
        own_state = self.state_dict()

        for name, param in pm.items():

            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            if name in own_state:
                if own_state[name].data.shape != param.shape:
                    continue
                own_state[name].copy_(param)
                cnt += 1

        print('%d parameters loaded successfully' % cnt)

    def get_other_model_to_load_from(self):
        """ Returns the model from which we are going to load the parameters for initialization.
        Default is the same model as we are using. """
        return self

    def freeze_parameters(self):
        """ Freezes the parameters that are not going to be updated while training.
         Default is no freezing."""
        return
