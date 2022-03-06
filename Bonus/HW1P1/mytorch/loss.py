# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        batch_size=x.shape[0]
        self.batch_size=batch_size
        x1=np.zeros((batch_size, 10))
        for i in range(0, batch_size):
            sum=0
            a = np.max(x[i])
            for j in range(0, 10):
                sum+=np.exp(x[i][j]-a)
            x1[i] = np.exp(x[i])/np.exp(a+np.log(sum))
        celoss = np.zeros((batch_size,))
        for i in range(0, batch_size):
            for j in range(0, 10):
                celoss[i]-=y[i][j]*np.log(x1[i][j])
        self.x1 = x1
        return celoss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        
#        batch_size = self.batch_size
#        derivative = np.zeros((batch_size, 10))
#        L_x1=np.zeros((batch_size,1, 10))
#        Jx1_x=np.zeros((batch_size,10, 10))
#        for i in range(0, batch_size):
#            for j in range(0, 10):
#                L_x1[i][0][j]=-self.labels[i][j]/self.x1[i][j]
#                for k in range(0, 10):
#                    if(j==k):
#                        Jx1_x[i][j][k]=self.x1[i][j]*(1-self.x1[i][j])
#                    else:
#                        Jx1_x[i][j][k]=-self.x1[i][j]*self.x1[i][k]
#            derivative[i] = np.dot(L_x1[i], Jx1_x[i])
        return self.x1-self.labels

class L2(Criterion):
    """
    L2 loss
    """

    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.x = x
        self.labels = y
        batch_size=x.shape[0]
        self.batch_size=batch_size
        l2loss=np.zeros((batch_size, 10))
        for i in range(0, batch_size):
            for j in range(0, 10):
               l2loss[i]+=0.5*(x[i][j]-y[i][j])**2
        return l2loss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        batch_size = self.batch_size
        derivative = np.zeros((batch_size, 10))
        derivative=self.x-self.labels
        return derivative
