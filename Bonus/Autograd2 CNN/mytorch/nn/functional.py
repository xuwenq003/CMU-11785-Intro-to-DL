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
    

def conv1d_backward(delta, x, W, b, stride):
    # Hint: x is input, W is weight, b is bias
    
#    print(stride)
    stride = int(stride)
#    print(stride)
    batch_size = delta.shape[0]
    output_size = delta.shape[2]
    input_size = x.shape[2]
    kernel_size = W.shape[2]
    out_channel = W.shape[0]
    in_channel = W.shape[1]
    
    # db
    db = np.sum(delta, axis=(0,2))
    
    # dW
    dW = np.zeros_like(W)
    for i in range(kernel_size):
        dW[:,:,i] = np.tensordot(delta, x[:,:,i:i+stride*output_size:stride],axes=([0,2],[0,2]))
    
    # dx
    dx = np.zeros((batch_size, in_channel, input_size))
    delta_dialated = np.zeros((batch_size, out_channel, input_size-kernel_size+1))
    delta_dialated[:,:,range(0,input_size-kernel_size+1,stride)] = delta
    delta_padded = np.pad(delta_dialated, ((0,0),(0,0),(kernel_size-1, kernel_size-1)), 'constant', constant_values=0)
    W_flipped = np.flip(W,2)
    for i in range(input_size):
        dx[:,:,i] = np.tensordot(delta_padded[:,:,i:i+kernel_size], W_flipped, axes=([1,2],[0,2]))
    
    return dx, dW, db, 1


def conv2d_backward(delta, x, W, b, stride):
    
    stride = int(stride)
    batch_size = delta.shape[0]
    
    output_width = delta.shape[2]
    output_height = delta.shape[3]
    
    input_width = x.shape[2]
    input_height = x.shape[3]
    
    kernel_size = W.shape[2]
    
    out_channel = W.shape[0]
    in_channel = W.shape[1]
    

    # db
    db = np.sum(delta, axis=(0,2,3))
        
    # dW
    dW = np.zeros_like(W)
    for i in range(kernel_size):
        for j in range(kernel_size):
            dW[:,:,i,j] = np.tensordot(delta, x[:,:,i:i+stride*output_width:stride,j:j+stride*output_height:stride],axes=([0,2,3],[0,2,3]))
        
    # dx
    dx = np.zeros((batch_size, in_channel, input_width, input_height))
    delta_dialated = np.zeros((batch_size, out_channel, input_width-kernel_size+1,input_height-kernel_size+1))
    delta_dialated[:,:,0:input_width-kernel_size+1:stride,0:input_height-kernel_size+1:stride] = delta
    delta_padded = np.pad(delta_dialated, ((0,0),(0,0),(kernel_size-1, kernel_size-1),(kernel_size-1, kernel_size-1)), 'constant', constant_values=0)
    W_flipped = np.flip(W,(2,3))
    for i in range(input_width):
        for j in range(input_height):
            dx[:,:,i,j] = np.tensordot(delta_padded[:,:,i:i+kernel_size,j:j+kernel_size], W_flipped, axes=([1,2,3],[0,2,3]))
        
    return dx, dW, db, 1


def flatten_backward(grad_output, a):
    raise NotImplementedError



