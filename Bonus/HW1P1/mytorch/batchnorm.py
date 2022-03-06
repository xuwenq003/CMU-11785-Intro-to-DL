# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from loss import L2
class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        self.x = x
        self.batch_size=x.shape[0]
        if eval:
            self.norm = (x-self.running_mean)/np.sqrt(self.running_var+self.eps)
        else:
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            self.norm = (x-self.mean)/np.sqrt(self.var+self.eps)
            # Update running batch statistics
            self.running_mean = self.alpha*self.running_mean+(1-self.alpha)*self.mean
            self.running_var = self.alpha*self.running_var+(1-self.alpha)*self.var

        self.out = self.gamma*self.norm+self.beta

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        b = self.batch_size
        sqrt_var_eps = np.sqrt(self.var+self.eps)
        self.dgamma = np.sum(self.norm*delta, axis=0, keepdims=True)
        self.dbeta = np.sum(delta, axis=0, keepdims=True)
        dnorm = self.gamma*delta
        dvar = -0.5*(np.sum((dnorm*(self.x-self.mean))/sqrt_var_eps**3, axis=0))
        first_term_dmu = -(np.sum(dnorm/sqrt_var_eps, axis=0))
        second_term_dmu = -(2/b)*dvar*(np.sum(self.x-self.mean, axis=0))
        dmu = first_term_dmu+second_term_dmu
        first_term_dx = dnorm/sqrt_var_eps
        second_term_dx = dvar*(2/b)*(self.x-self.mean)
        third_term_dx = dmu*(1/b)
        return first_term_dx+second_term_dx+third_term_dx
