# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

"""
In the linear.py file, attributes have been added to the Linear class to make
implementing Adam easier, check them out!

self.mW = np.zeros(None) #mean derivative for W
self.vW = np.zeros(None) #squared derivative for W
self.mb = np.zeros(None) #mean derivative for b
self.vb = np.zeros(None) #squared derivative for b
"""

class adam():
    def __init__(self, model, beta1=0.9, beta2=0.999, eps = 1e-8):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = self.model.lr
        self.t = 0 # Number of Updates

    def step(self):
        '''
        * self.model is an instance of your MLP in hw1/hw1.py, it has access to
          the linear layer's list.
        * Each linear layer is an instance of the Linear class, and thus has
          access to the added class attributes dicussed above as well as the
          original attributes such as the weights and their gradients.
        '''
        self.t += 1
        
        # Add your code here!
        for ll in self.model.linear_layers:
            ll.mW = self.beta1*ll.mW+(1-self.beta1)*ll.dW
            ll.mb = self.beta1*ll.mb+(1-self.beta1)*ll.db
            ll.vW = self.beta2*ll.vW+(1-self.beta2)*ll.dW*ll.dW
            ll.vb = self.beta2*ll.vb+(1-self.beta2)*ll.db*ll.db
            
            mW_hat = ll.mW/(1-self.beta1**self.t)
            mb_hat = ll.mb/(1-self.beta1**self.t)
            vW_hat = ll.vW/(1-self.beta2**self.t)
            vb_hat = ll.vb/(1-self.beta2**self.t)
            
            ll.W-=self.lr*mW_hat/np.sqrt(vW_hat+self.eps)
            ll.b-=self.lr*mb_hat/np.sqrt(vb_hat+self.eps)
            
class adamW():
    def __init__(self, model, beta1=0.9, beta2=0.999, eps = 1e-8, lbd=1e-2):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lbd = lbd
        self.lr = self.model.lr
        self.t = 0 # Number of Updates

    def step(self):
        '''
        * self.model is an instance of your MLP in hw1/hw1.py, it has access to
          the linear layer's list.
        * Each linear layer is an instance of the Linear class, and thus has
          access to the added class attributes dicussed above as well as the
          original attributes such as the weights and their gradients.
        '''
        self.t += 1
        
        # Add your code here!
        for ll in self.model.linear_layers:
            ll.W-=ll.W*self.lr*self.lbd
            ll.b-=ll.b*self.lr*self.lbd
            
            ll.mW = self.beta1*ll.mW+(1-self.beta1)*ll.dW
            ll.mb = self.beta1*ll.mb+(1-self.beta1)*ll.db
            ll.vW = self.beta2*ll.vW+(1-self.beta2)*ll.dW*ll.dW
            ll.vb = self.beta2*ll.vb+(1-self.beta2)*ll.db*ll.db
            
            mW_hat = ll.mW/(1-self.beta1**self.t)
            mb_hat = ll.mb/(1-self.beta1**self.t)
            vW_hat = ll.vW/(1-self.beta2**self.t)
            vb_hat = ll.vb/(1-self.beta2**self.t)
            
            ll.W-=self.lr*mW_hat/np.sqrt(vW_hat+self.eps)
            ll.b-=self.lr*mb_hat/np.sqrt(vb_hat+self.eps)
