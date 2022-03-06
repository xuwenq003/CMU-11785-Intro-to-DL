import numpy as np
from mytorch.nn.functional import *

class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self, autograd_engine):
        self.state = None
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self, autograd_engine):
        super(Identity, self).__init__(autograd_engine)

    def forward(self, x):
        self.state = x
        return self.state

class Sigmoid(Activation):
    def __init__(self, autograd_engine):
        super(Sigmoid, self).__init__(autograd_engine)

    def forward(self, x):
        z1 = np.zeros_like(x)-x
        self.autograd_engine.add_operation([np.zeros_like(x),x],z1,[None,None],sub_backward)
        
        z2 = np.exp(z1)
        self.autograd_engine.add_operation([z1],z2,[None],exp_backward)

        z3 = np.ones_like(z2)+z2
        self.autograd_engine.add_operation([np.ones_like(z2),z2],z3,[None,None],add_backward)

        z4 = np.ones_like(z3)/z3
        self.autograd_engine.add_operation([np.ones_like(z3),z3],z4,[None,None],div_backward)
        
        self.state = z4
        return self.state

class Tanh(Activation):
    def __init__(self, autograd_engine):
        super(Tanh, self).__init__(autograd_engine)

    def forward(self, x):
        z0 = np.ones_like(x)*2*x
        self.autograd_engine.add_operation([np.ones_like(x)*2,x],z0,[None,None],mul_backward)
        
        z1 = np.exp(z0)
        self.autograd_engine.add_operation([z0],z1,[None],exp_backward)
        
        z2 = np.ones_like(z1)+z1
        self.autograd_engine.add_operation([np.ones_like(z1),z1],z2,[None,None],add_backward)
        
        z3 = 2*np.ones_like(z2)/z2
        self.autograd_engine.add_operation([2*np.ones_like(z2),z2],z3,[None,None],div_backward)
        
        z4 = np.ones_like(z3)-z3
        self.autograd_engine.add_operation([np.ones_like(z3), z3],z4,[None,None],sub_backward)
        
        self.state = z4
        return self.state

class ReLU(Activation):
    def __init__(self, autograd_engine):
        super(ReLU, self).__init__(autograd_engine)

    def forward(self, x):
        self.state = (np.abs(x)+x)/2
        self.autograd_engine.add_operation([x],self.state,[None],relu_backward)
        return self.state
