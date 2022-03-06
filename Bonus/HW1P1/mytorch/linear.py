# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear(object):
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        # TODO: Complete these but do not change the names.
        self.dW = np.zeros((in_feature, out_feature))
        self.db = np.zeros(out_feature)

        self.momentum_W = np.zeros((in_feature, out_feature))
        self.momentum_b = np.zeros(out_feature)

        #ADAM:
        self.mW = np.zeros((in_feature, out_feature)) #mean derivative for W
        self.vW = np.zeros((in_feature, out_feature))  #squared derivative for W
        self.mb = np.zeros(out_feature) #mean derivative for b
        self.vb = np.zeros(out_feature)  #squared derivative for b

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.x = x
        self.out = np.matmul(x, self.W) + self.b
        return self.out

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """

        self.dW = np.matmul(self.x.T, delta) / (self.x.shape[0])
        self.db = np.mean(delta, axis=0)
        self.dx = np.matmul(delta, self.W.T)
        out = self.dx
        return out
