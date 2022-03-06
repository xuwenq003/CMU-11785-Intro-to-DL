import numpy as np

class MaxPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        batch_size, in_channel, input_width, input_height = x.shape
        self.batch_size = batch_size
        self.in_channel = in_channel
        self.input_width = input_width
        self.input_height = input_height
        output_width = (input_width-self.kernel)//self.stride+1
        output_height = (input_height-self.kernel)//self.stride+1
        out = np.zeros((batch_size, in_channel, output_width, output_height))
        self.max_idx = np.zeros((batch_size, in_channel, output_width, output_height),dtype=int)
        for i in range(batch_size):
            for j in range(in_channel):
                for k in range(output_width):
                    for l in range(output_height):
                        idx = np.argmax(x[i,j,k*self.stride:k*self.stride+self.kernel,l*self.stride:l*self.stride+self.kernel])
                        self.max_idx[i,j,k,l] = idx
                        out[i,j,k,l] = np.max(x[i,j,k*self.stride:k*self.stride+self.kernel,l*self.stride:l*self.stride+self.kernel])
        return out

    
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        dx = np.zeros((self.batch_size,self.in_channel,self.input_width,self.input_height))
        output_width, output_height = delta.shape[2], delta.shape[3]
        for i in range(self.batch_size):
            for j in range(self.in_channel):
                for k in range(output_width):
                    for l in range(output_height):
                        idx_x = self.max_idx[i,j,k,l]//self.kernel
                        idx_y = self.max_idx[i,j,k,l]%self.kernel
                        dx[i,j,k*self.stride+idx_x,l*self.stride+idx_y] = delta[i][j][k][l]
        return dx

class MeanPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        batch_size, in_channel, input_width, input_height = x.shape
        self.batch_size = batch_size
        self.in_channel = in_channel
        self.input_width = input_width
        self.input_height = input_height
        output_width = (input_width-self.kernel)//self.stride+1
        output_height = (input_height-self.kernel)//self.stride+1
        out = np.zeros((batch_size, in_channel, output_width, output_height))
        for i in range(batch_size):
            for j in range(in_channel):
                for k in range(output_width):
                    for l in range(output_height):
                        out[i,j,k,l] = np.mean(x[i,j,k*self.stride:k*self.stride+self.kernel,l*self.stride:l*self.stride+self.kernel])
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        
        dx = np.zeros((self.batch_size,self.in_channel,self.input_width,self.input_height))
        output_width, output_height = delta.shape[2], delta.shape[3]
        for i in range(self.batch_size):
            for j in range(self.in_channel):
                for k in range(output_width):
                    for l in range(output_height):
                        dx[i,j,k*self.stride:k*self.stride+self.kernel,l*self.stride:l*self.stride+self.kernel] += delta[i][j][k][l]/(self.kernel**2)
        return dx
