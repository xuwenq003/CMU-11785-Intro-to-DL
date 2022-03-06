import numpy as np
from mytorch.nn.functional import conv1d_backward, conv2d_backward, flatten_backward


class Conv1d():
    def __init__(self, input_channels, output_channels, kernel_size, stride, autograd_engine, bias=True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride= stride
        self.bias = bias
        self.autograd_engine = autograd_engine

        self.W = np.random.uniform

        self.W = np.random.uniform(-1, 1, size=(self.output_channels, self.input_channels, self.kernel_size)).astype(np.float64)
        self.b = np.zeros((self.output_channels)).astype(np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        # TODO: Remember to use add_operation to record these operations in
        #      the autograd engine after each operation

        # TODO: remember to return the computed value
        batch_size = x.shape[0]
        input_size = x.shape[2]
        output_size = (input_size-self.kernel_size)//self.stride+1
        self.x = x
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        
        out = np.zeros((batch_size, self.output_channels, output_size))
        b = np.zeros((batch_size, self.output_channels))
        for i in range(batch_size):
            b[i] = self.b.reshape(1,self.output_channels)
        for i in range(output_size):
            out[:,:,i]=np.tensordot(x[:,:,i*self.stride:i*self.stride+self.kernel_size], self.W, axes =([1,2],[1,2]))+b
        self.autograd_engine.add_operation([x, self.W, self.b, np.array([self.stride])], out, [None, self.dW, self.db, None], conv1d_backward)
        return out
#        raise NotImplementedError

class Conv2d():
    def __init__(self, input_channels, output_channels, kernel_size, stride, autograd_engine, bias=True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.autograd_engine = autograd_engine

        self.W = np.random.uniform(-1, 1, size=(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)).astype(np.float64)
        self.b = np.zeros((self.output_channels)).astype(np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        # TODO: Remember to use add_operation to record these operations in
        #      the autograd engine after each operation
    
        # TODO: remember to return the computed value
        batch_size = x.shape[0]
        input_width = x.shape[2]
        input_height = x.shape[3]
        output_width = (input_width-self.kernel_size)//self.stride+1
        output_height = (input_height-self.kernel_size)//self.stride+1
        self.x = x
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height
        
        out = np.zeros((batch_size, self.output_channels, output_width, output_height))
        b = np.zeros((batch_size, self.output_channels))
        for i in range(batch_size):
            b[i] = self.b.reshape(1,self.output_channels)
        for i in range(output_width):
            for j in range(output_height):
                out[:,:,i,j]=np.tensordot(x[:,:,i*self.stride:i*self.stride+self.kernel_size,j*self.stride:j*self.stride+self.kernel_size], self.W, axes =([1,2,3],[1,2,3]))+b
        self.autograd_engine.add_operation([x, self.W, self.b, np.array([self.stride])], out, [None, self.dW, self.db, None], conv2d_backward)
        return out
#        raise NotImplementedError


class Flatten():

    def __init__(self, autograd_engine):
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        raise NotImplementedError

