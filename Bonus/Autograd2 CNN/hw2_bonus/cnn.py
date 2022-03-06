import numpy as np

import sys
sys.path.append('..')
import os

from mytorch import autograd_engine
import mytorch.nn as nn
from mytorch.nn.functional import *

DATA_PATH = "./data"


class CNN(object):
    """
    A simple convolutional neural network
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides, num_linear_neurons,
                 activations, criterion, lr, autograd_engine):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """
        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations

        self.criterion = criterion

        self.lr = lr

        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        ## Your code goes here -->
        # self.convolutional_layers (list Conv1D) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)
        # <---------------------
        # input_channels, output_channel, kernel_size, stride, autograd_engine, bias

        num_channels = [num_input_channels] + num_channels

        self.convolutional_layers = None

        self.flatten = nn.Flatten(autograd_engine)

        self.autograd_engine = autograd_engine # NOTE: Use this Autograd object for backward

        cnn_out_neurons = input_width
        for i in range(len(num_channels)):
            cnn_out_neurons = (cnn_out_neurons - kernel_sizes[i]) // strides[i] + 1
        cnn_out_neurons *= num_linear_neurons[-1]
        self.linear_layer = nn.Linear(cnn_out_neurons, num_linear_neurons, self.autograd_engine)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, num_input_channels, input_width)
        Return:
            out (np.array): (batch_size, num_linear_neurons)
        """
        ## Your code goes here -->
        # Iterate through each layer
        # <---------------------
        raise NotImplementedError

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear layers.
        raise NotImplementedError


    def step(self):
        # Apply a step to the weights and biases of the layers.
        raise NotImplementedError


    def backward(self, labels):
        # The magic of autograd: This is 2 lines.
        # Get the loss.
        # Call autograd backward.
        # or self.criterion.backward()
        raise NotImplementedError


    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        # NOTE: Put the inputs in the correct order for the criterion
        # return self.criterion().sum()
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(cnn, dset, nepochs, batch_size=1):
    # NOTE: Because the batch size is 1 (unless you support
    # broadcasting) the cnn training will be slow.
    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...
    for e in range(nepochs):

        # Per epoch setup ...
        for b in range(0, len(trainx)):
            # Train ...
            # NOTE: Batchsize is 1 for this bonus unless you support
            # broadcasting/unbroadcasting then you can change this in
            # the mlp_runner.py
            x = np.expand_dims(trainx[b], 0)
            y = np.expand_dims(trainy[b], 0)

        for b in range(0, len(valx)):
            # Val ...
            x = np.expand_dims(valx[b], 0)
            y = np.expand_dims(valy[b], 0)

        # Accumulate data...

    # Cleanup ...

    # Return results ...
    return (training_losses, training_errors, validation_losses, validation_errors)


def load_data():
    train_x = np.load(os.path.join(DATA_PATH, "train_data.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    val_x = np.load(os.path.join(DATA_PATH, "val_data.npy"))
    val_y = np.load(os.path.join(DATA_PATH, "val_labels.npy"))

    train_x = train_x / 255
    val_x = val_x / 255

    return train_x, train_y, val_x, val_y


if __name__=='__main__':
    pass
    # np.random.seed(0)
    # ### Testing with random sample for now
    # train_x, train_y, val_x, val_y = load_data()
    # train = (train_x, train_y)
    # val = (val_x, val_y)
    #
    # epochs = 5
    # autograd = autograd_engine.Autograd()
    # input_width = 128
    # num_input_channels = 1
    # num_channels = [56, 28, 14]
    # kernel_sizes = [5, 6, 2]
    # strides = [1, 2, 2]
    # num_linear_neurons = 10
    # activations = [nn.Tanh(autograd_engine), nn.ReLU(autograd_engine), nn.Sigmoid(autograd_engine)]
    # criterion = nn.loss.SoftmaxCrossEntropy(autograd)
    # lr = 0.001
    #
    # cnn = CNN(input_width, num_input_channels, num_channels, kernel_sizes, strides, num_linear_neurons,
    #           activations, criterion, lr, autograd_engine)
    #
    # train_loss, train_error, valid_loss, valid_error = get_training_stats(cnn, train, val, epochs)
