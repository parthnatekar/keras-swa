""" Keras SWA: callback utility for performing stochastic weight averaging (SWA).
"""

import keras.backend as K
from keras.callbacks import Callback
from keras.layers import BatchNormalization
import json
import numpy as np

class BN_Update(Callback):
    """ Stochastic Weight Averging.

    # Paper
        title: Averaging Weights Leads to Wider Optima and Better Generalization
        link: https://arxiv.org/abs/1803.05407

    # Arguments
        start_epoch:   integer, epoch when swa should start.
        lr_schedule:   string, type of learning rate schedule.
        swa_lr:        float, learning rate for swa.
        swa_lr2:       float, upper bound of cyclic learning rate.
        swa_freq:      integer, length of learning rate cycle.
        batch_size     integer, batch size (for batch norm with generator)
        verbose:       integer, verbosity mode, 0 or 1.
    """
    def __init__(self,
                 start_epoch,
                 swa_weights,
                 batch_size=None,
                 verbose=0,
                 checkpoint_save_path=None):

        super(BN_Update, self).__init__()
        self.start_epoch = start_epoch - 1
        self.verbose = verbose
        self.checkpoint_save_path = checkpoint_save_path

        if self.params.get('epochs') > 1:
            raise ValueError('"total epochs" must be 1 for batch norm update run')

    def on_train_begin(self, logs=None):

        self._check_batch_norm()

    def on_epoch_begin(self, epoch, logs=None):

        self.is_batch_norm_epoch = epoch == self.epochs - 1 and self.has_batch_norm

        if self.is_batch_norm_epoch:

            K.set_value(self.model.optimizer.lr, 0)

            self.model.set_weights(swa_weights)

            if self.verbose > 0:
                    print('\nEpoch %05d: reinitializing batch normalization layers' 
                        % (epoch + 1))
                
            self._reset_batch_norm()

            if self.verbose > 0:
                print('\nEpoch %05d: running forward pass to adjust batch normalization'
                      % (epoch + 1))


    def on_train_end(self, logs=None):

        self._restore_batch_norm()

    def _check_batch_norm(self):

        self.batch_norm_momentums = []
        self.batch_norm_layers = []
        self.has_batch_norm = False
        self.running_bn_epoch = False

        for layer in self.model.layers:
            if issubclass(layer.__class__, BatchNormalization):
                self.has_batch_norm = True
                self.batch_norm_momentums.append(layer.momentum)
                self.batch_norm_layers.append(layer)

        if self.verbose > 0 and self.has_batch_norm:
            print('Model uses batch normalization. SWA will require last epoch '
                  'to be a forward pass and will run with no learning rate')

    def _reset_batch_norm(self):

        for layer in self.batch_norm_layers:

            # to get properly initialized moving mean and moving variance weights
            # we initialize a new batch norm layer from the config of the existing
            # layer, build that layer, retrieve its reinitialized moving mean and 
            # moving var weights and then delete the layer
            bn_config = layer.get_config()
            new_batch_norm = BatchNormalization(**bn_config)
            new_batch_norm.build(layer.input_shape)
            new_moving_mean, new_moving_var = new_batch_norm.get_weights()[-2:]
            # get rid of the new_batch_norm layer
            del new_batch_norm
            # get the trained gamma and beta from the current batch norm layer
            trained_weights = layer.get_weights()
            new_weights = []
            # get gamma if exists
            if bn_config['scale']:
                new_weights.append(trained_weights.pop(0))
            # get beta if exists
            if bn_config['center']:
                new_weights.append(trained_weights.pop(0))
            new_weights += [new_moving_mean, new_moving_var]
            # set weights to trained gamma and beta, reinitialized mean and variance
            layer.set_weights(new_weights)
          
    def _restore_batch_norm(self):

        for layer, momentum in zip(self.batch_norm_layers, self.batch_norm_momentums):
            layer.momentum = momentum