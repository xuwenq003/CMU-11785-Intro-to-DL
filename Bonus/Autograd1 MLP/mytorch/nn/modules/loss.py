import numpy as np
from mytorch.nn.functional import *

class MSELoss:
    def __init__(self, autograd_engine):
        self.autograd_engine = autograd_engine
        self.loss_val = None

    def __call__(self, y, y_hat):
        self.forward(y, y_hat)

    # TODO: Use your working MSELoss forward and add operations to 
    # autograd_engine.
    def forward(self, y, y_hat):
        """
            This class is similar to the wrapper functions for the activations
            that you wrote in functional.py with a couple of key differences:
                1. Notice that instead of passing the autograd object to the forward
                    method, we are instead saving it as a class attribute whenever
                    an MSELoss() object is defined. This is so that we can directly 
                    call the backward() operation on the loss as follows:
                        >>> mse_loss = MSELoss(autograd_object)
                        >>> mse_loss(y, y_hat)
                        >>> mse_loss.backward()

                2. Notice that the class has an attribute called self.loss_val. 
                    You must save the calculated loss value in this variable and 
                    the forward() function is not expected to return any value.
                    This is so that we do not explicitly pass the divergence to 
                    the autograd engine's backward method. Rather, calling backward()
                    on the MSELoss object will take care of that for you.

            Args:
                - y (np.ndarray) : the ground truth,
                - y_hat (np.ndarray) : the output computed by the network,

            Returns:
                - No return required
        """
        #TODO: Use the primitive operations to calculate the MSE Loss
        #TODO: Remember to use add_operation to record these operations in
        #      the autograd engine after each operation

    def backward(self):
        # You can call autograd's backward here or in the mlp.
        raise NotImplementedError

# Hint: To simplify things you can just make a backward for this loss and not
# try to do it for every operation.
class SoftmaxCrossEntropy:
    def __init__(self, autograd_engine):
        self.loss_val = None
        self.y_grad_placeholder = None
        self.autograd_engine = autograd_engine

    def __call__(self, y, y_hat):
        return self.forward(y, y_hat)

    def forward(self, y, y_hat):
        """
            Refer to the comments in MSELoss
        """
        
        y_hat_exp = np.exp(y_hat)
        y_hat_exp_sum = np.sum(y_hat_exp)
        y_hat_soft = y_hat_exp/y_hat_exp_sum
        loss = -np.sum(y*np.log(y_hat_soft))
#        print(loss)
        self.autograd_engine.add_operation([y_hat, y], loss, [None, None], SoftmaxCrossEntropy_backward)
        
        self.loss_val = loss
        
        return np.array([self.loss_val])

    def backward(self):
        # You can call autograd's backward here OR in the mlp.
        self.autograd_engine.backward(self.loss_val)
#        raise NotImplementedError
