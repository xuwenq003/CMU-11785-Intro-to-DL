# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
    # Dropout probability
        self.p = p
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x, train = True):
        if train==False:
            return x
        else:
            self.mask = np.random.binomial(1, self.p, x.shape)
            out = x*self.mask/self.p
            return out
            
    def backward(self, delta):
    # 1) This method is only called during training.
        return delta*self.mask
