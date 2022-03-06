# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, train=True):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          train (boolean): whether the model is in training mode
        Return:
          np.array of same shape as input x
        """
        # 1) Get and apply a per-channel mask generated from np.random.binomial
        # 2) Scale your output accordingly
        # 3) During test time, you should not apply any mask or scaling.
        if not train:
            return x
        else:
            self.mask = np.zeros((x.shape[0],x.shape[1]))
            out = x
            for i in range(x.shape[0]):
                self.mask[i] = np.random.binomial(1, 1-self.p, x.shape[1])
                for j in range(x.shape[1]):
                    if(self.mask[i][j]==0):
                        out[i][j]=0
            return out/(1-self.p)
                

#        raise NotImplementedError

    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        # 1) This method is only called during training.
        derivative = delta
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                if(self.mask[i][j]==0):
                    derivative[i][j]=0
        return derivative/(1-self.p)

