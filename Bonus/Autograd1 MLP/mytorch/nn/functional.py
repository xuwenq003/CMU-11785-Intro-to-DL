import numpy as np
from mytorch.autograd_engine import Autograd

'''
Mathematical Functionalities
    These are some IMPORTANT things to keep in mind:
    - Make sure grad of inputs are exact same shape as inputs.
    - Make sure the input and output order of each function is consistent with
        your other code.
    Optional:
    - You can account for broadcasting, but it is not required 
        in the first bonus.
''' 
def add_backward(grad_output, a, b):
    a_grad = grad_output
    b_grad = grad_output
    return a_grad, b_grad

def sub_backward(grad_output, a, b):
    a_grad = grad_output
    b_grad = -grad_output
    return a_grad, b_grad

def matmul_backward(grad_output, a, b):
    a_grad = np.dot(grad_output, b.T)
    b_grad = np.dot(a.T, grad_output)
    return a_grad, b_grad

def mul_backward(grad_output, a, b):
    a_grad = b*grad_output
    b_grad = a*grad_output
    return a_grad, b_grad

def div_backward(grad_output, a, b):
    a_grad = 1/b*grad_output
    b_grad = -a/b/b*grad_output
    return a_grad, b_grad

def log_backward(grad_output, a):
    a_grad = 1/a*grad_output
    return a_grad

def exp_backward(grad_output, a):
    a_grad = np.exp(a)*grad_output
    return a_grad

def relu_backward(grad_output, a):
    a_grad = np.sign((np.abs(a)+a)/2)*grad_output
    return a_grad

def max_backward(grad_output, a):
    pass


def SoftmaxCrossEntropy_backward(grad_output, y_hat, y):
    """
    NOTE: Since the gradient of the Softmax CrossEntropy Loss is
          is straightforward to compute, you may choose to implement
          this directly rather than rely on the backward functions of
          more primitive operations.
    """
    y_hat_exp = np.exp(y_hat)
    y_hat_exp_sum = np.sum(y_hat_exp)
    y_hat_soft = y_hat_exp/y_hat_exp_sum
    y_hat_grad = y_hat_soft-y
    y_grad = 1
    return y_hat_grad, y_grad
    
