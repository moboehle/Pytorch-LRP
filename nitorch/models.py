import numpy as np
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from nitorch.data import show_brain


class _CAE_3D(nn.Module):
    '''
    Parent Convolutional Autoencoder class for 3D images. 
    All other Convolutional Autoencoder classes must inherit from this class.
    '''
    def __init__(self, conv_channels):        
        super().__init__()        
        # check if there are multiple convolution layers within a layer of the network or not
        self.is_nested_conv = ([isinstance(each_c, (list, tuple)) for each_c in conv_channels])
        if(any(self.is_nested_conv) and not all(self.is_nested_conv)):
             raise TypeError(" `conv_channels` can't be a mixture of both lists and ints.")
        self.is_nested_conv = any(self.is_nested_conv)
        
        self.layers = len(conv_channels)
        self.conv_channels = self._format_channels(conv_channels, self.is_nested_conv)
        self.valid_activations = {'ELU': nn.ELU, 'HARDSHRINK': nn.Hardshrink, 'HARDTANH': nn.Hardtanh,
 'LEAKYRELU':nn.LeakyReLU, 'LOGSIGMOID': nn.LogSigmoid, 'PRELU':nn.PReLU, 'RELU':nn.ReLU, 'RELU6': nn.ReLU6, 
 'RRELU': nn.RReLU, 'SELU': nn.SELU, 'SIGMOID': nn.Sigmoid, 'SOFTPLUS': nn.Softplus, 
 'SOFTSHRINK': nn.Softshrink, 'TANH': nn.Tanh, 'TANHSHRINK': nn.Tanhshrink, 'THRESHOLD': nn.Threshold}

    def _format_channels(self, conv_channels, is_nested_conv = False):
        channels = []
        if(is_nested_conv):
            for i in range(len(conv_channels)):
                    inner_channels = []
                    for j in range(len(conv_channels[i])):
                        if (i == 0) and (j == 0):
                            inner_channels.append([1, conv_channels[i][j]])
                        elif (j == 0) :
                            inner_channels.append([conv_channels[i-1][-1], conv_channels[i][j]])
                        else:
                            inner_channels.append([conv_channels[i][j-1], conv_channels[i][j]])
                    channels.append(inner_channels)
        else:
            for i in range(len(conv_channels)):
                if (i == 0):
                    channels.append([1, conv_channels[i]])
                else:
                    channels.append([conv_channels[i-1], conv_channels[i]])

        return channels


    def assign_parameter(self, parameter, param_name, enable_nested = True):
        ''' Wrapper for parameters of the Autoencoder. 
        Checks if the len and type of the parameter is acceptable.
        If the parameter is just an single value,
        makes its length equal to the number of layers defined in conv_channels
        '''
        if(isinstance(parameter, (int, str))):
            if(self.is_nested_conv and enable_nested):
                return_parameter = [len(inner_list)*[parameter] for inner_list in self.conv_channels]
            else:
                return_parameter = (self.layers * [parameter])
        # Perform sanity checks if a list is already provided 
        elif(isinstance(parameter, (list, tuple))):
            if(len(parameter) != self.layers): 
                raise ValueError("The parameter '{}' can either be a single int \
or must be a list of the same length as 'conv_channels'.".format(
        param_name))
        
            if(self.is_nested_conv and enable_nested):
                if(any(
                    [len(c) != len(p) for c, p in zip(self.conv_channels, parameter)]
                    )):
                    raise ValueError("The lengths of the inner lists of the parameter {} \
have to be same as the 'conv_channels'".format(param_name)) 
            # if all length checks pass just return the parameter
            return_parameter = parameter
            
        else: 
            raise TypeError("Parameter {} is neither an int/ valid str nor a list/tuple but is of type {}".format(
                param_name, parameter))

        return return_parameter


    def add_conv_with_activation(self, inp_channels, out_channels, kernel_size, padding, stride, activation_fn):
        node = nn.Sequential(
            nn.Conv3d(inp_channels, out_channels, kernel_size, padding = padding, stride = stride),
            self.valid_activations[activation_fn](inplace=True))        
        return node
        

    def add_deconv_with_activation(self, inp_channels, out_channels, kernel_size, padding, stride, out_padding, activation_fn):
        node = nn.Sequential(
            nn.ConvTranspose3d(inp_channels, out_channels, kernel_size
                , padding = padding, stride = stride, output_padding=out_padding),
            self.valid_activations[activation_fn](inplace=True))        
        return node
        
        
    def add_pool(self, pool_type, kernel_size, padding, stride):
        if(pool_type == "max"):
            node = nn.MaxPool3d(kernel_size, 
                        padding = padding, 
                        stride = stride, 
                        return_indices = True)
        elif(pool_type == "avg"):
            node = nn.AvgPool3d(kernel_size, 
                        padding = padding, 
                        stride = stride)
        else:
            raise TypeError("Invalid value provided for `pool_type`.\
Allowed values are `max`, `avg`.")
            
        return node
        
        
    def add_unpool(self, pool_type, kernel_size, padding, stride):
        if(pool_type == "max"):
            node = nn.MaxUnpool3d(kernel_size, 
                        padding = padding, 
                        stride = stride)
        elif(pool_type == "avg"):
            node = nn.MaxPool3d(kernel_size, 
                        padding = padding, 
                        stride = stride)
        else:
            raise TypeError("Invalid value provided for `pool_type`.\
Allowed values are `max`, `avg`.")
            
        return node
    
        
    def nested_reverse(self, mylist):
        result = []
        for e in mylist:
            if isinstance(e, (list, tuple)):
                result.append(self.nested_reverse(e))
            else:
                result.append(e)
        result.reverse()
        return result

    
class CAE_3D(_CAE_3D):
    '''
    3D Convolutional Autoencoder model with only convolution layers. Strided convolution
    can be used for undersampling. 
    '''
    def __init__(self
        , conv_channels
        , activation_fn = "RELU"
        , conv_kernel = 3
        , conv_padding = 1
        , conv_stride = 1
        , deconv_out_padding = None
        , second_fc_decoder = []
        ):
        '''
        Args:
            conv_channels : A list that defines the number of channels of each convolution layer.
            The length of the list defines the number of layers in the encoder. 
            The decoder is automatically constructed as an exact reversal of the encoder architecture.
            
            activation_fn (optional):  The non-linear activation function that will be appied after every layer
            of convolution / deconvolution. 
            Supported values {'ELU', 'HARDSHRINK', 'HARDTANH', 'LEAKYRELU', 'LOGSIGMOID', 'PRELU', 'RELU', 
            'RELU6', 'RRELU', 'SELU', 'SIGMOID', 'SOFTPLUS', 'SOFTSHRINK', 'TANH', 'TANHSHRINK', 'THRESHOLD'}
            By default nn.ReLu() is applied.
            Can either be a a single int (in which case the same activation is applied to all layers) or 
            a list of same length and shape as `conv_channels`.
            
            conv_kernel (optional): The size of the 3D convolutional kernels to be used. 
            Can either be a list of same length as `conv_channels` or a single int. In the
             former case each value in the list represents the kernel size of that particular
            layer and in the latter case all the layers are built with the same kernel size as 
            specified.

            conv_padding (optional): The amount of zero-paddings to be done along each dimension.
            Format same as conv_kernel.

            conv_stride (optional): The stride of the 3D convolutions.
            Format same as conv_kernel.

            deconv_out_padding (optional): The additional zero-paddings to be done to the output 
            of ConvTranspose / Deconvolutions in the decoder network.
            By default does (stride-1) number of padding.
            Format same as conv_kernel.
            
            second_fc_decoder (optional): By default this is disabled. 
            If a non-empty list of ints is provided then a secondary fully-connected decoder 
            network is constructed as per the list.
            Each value represents the number of cells in each layer. Just like `conv_channels`
            the length of the list defines the number of layers.
            If enabled, the forward() method returns a list of 2 outputs, one from the Autoencoder's
            decoder and the other from this fully-connected decoder network.            
        '''
        super().__init__(conv_channels)

        assert not(self.is_nested_conv), "The conv_channels must be a list of ints (i.e. number of channels).\
It cannot be a list of lists."

        self.conv_kernel = self.assign_parameter(conv_kernel, "conv_kernel")
        self.conv_padding = self.assign_parameter(conv_padding, "conv_kernel")
        self.conv_stride = self.assign_parameter(conv_stride, "conv_stride")
        if(deconv_out_padding == None):
            deconv_out_padding = [s-1 for s in self.conv_stride]
        self.deconv_out_padding = self.assign_parameter(deconv_out_padding, "deconv_out_padding")
        
        self.activation_fn = self.assign_parameter(activation_fn, "activation_function")

        for activation in self.activation_fn:
            assert activation.upper() in self.valid_activations.keys(), "activation functions can only be one of the following str :\n {}".format(
                    self.valid_activations.keys())
            
        # set the switches used in forward() as false  by default
        self.debug = False
        self.return_encoder_out = False
        
        if(second_fc_decoder):
            self.second_fc_decoder = self._format_channels(second_fc_decoder)[1:]
        else:
            self.second_fc_decoder = []

        self.convs = nn.ModuleList()
        self.deconvs = nn.ModuleList()
        
        for i in range(self.layers):
            # build the encoder
            self.convs.append(
                self.add_conv_with_activation(
                    self.conv_channels[i][0], self.conv_channels[i][1], 
                    self.conv_kernel[i]
                    , self.conv_padding[i]
                    , self.conv_stride[i]
                    , self.activation_fn[i]
                    )
                )
            # build the decoder
            self.deconvs.append(
                self.add_deconv_with_activation(
                    self.conv_channels[-i-1][1], self.conv_channels[-i-1][0], 
                    self.conv_kernel[-i-1]
                    , self.conv_padding[-i-1]
                    , self.conv_stride[-i-1]
                    , self.deconv_out_padding[-i-1]
                    , self.activation_fn[-i-1]
                    )
                )
        if(self.second_fc_decoder):
        # build the second fc decoder
            self.fcs = nn.ModuleList()
            for layer in self.second_fc_decoder:
                self.fcs.append(
                    nn.Linear(layer[0], layer[1])
                )
                
                
    def set_debug(self, bool_val):
        self.debug = bool_val
    
    def set_return_encoder_out(self, bool_val):
        self.return_encoder_out = bool_val        

    def forward(self, x):

            if(self.debug): print("\nImage dims ="+str(x.size()))

            #encoder
            for i, conv in enumerate(self.convs):
                x = conv(x)
                if(self.debug): print("conv{} output dim = {}".format(i+1, x.size()))
            
            encoder_out = x
            
            if(self.debug): print("\nEncoder output dims ="+str(encoder_out.size())+"\n")
            
            #decoder
            for i, deconv in enumerate(self.deconvs):
                x = deconv(x)
                if(self.debug): print("deconv{} output dim = {}".format(i+1, x.size()))
                        
            if(self.debug): print("\nDecoder output dims ="+str(x.size())+"\n")
            
            if(self.return_encoder_out):
                
                return [x, encoder_out]
            else:
                
                return x

            
            
class CAE_3D_with_pooling(_CAE_3D):
    '''
    3D Convolutional Autoencoder model with alternating Pooling layers. 
    '''
    def __init__(self
        , conv_channels
        , activation_fn = nn.ReLU
        , conv_kernel = 3, conv_padding = 1, conv_stride = 1
        , pool_type = "max"
        , pool_kernel = 2, pool_padding = 0, pool_stride = 2
        , deconv_out_padding = None
        ):
        '''
        Args:
            conv_channels : A nested list whose length defines the number of layers. Each layer
            can intern have multiple convolutions followed by a layer of Pooling. The lengths of the 
            inner list defines the number of convolutions per such layer and the value defines the number of
            channels for each of these convolutions.
            The decoder is constructed to be simply an exact reversal of the encoder architecture.
            
            activation_fn (optional):  The non-linear activation function that will be appied after every layer
            of convolution / deconvolution. By default nn.ReLu() is applied.
            Supported values {'ELU', 'HARDSHRINK', 'HARDTANH', 'LEAKYRELU', 'LOGSIGMOID', 'PRELU', 'RELU', 
            'RELU6', 'RRELU', 'SELU', 'SIGMOID', 'SOFTPLUS', 'SOFTSHRINK', 'TANH', 'TANHSHRINK', 'THRESHOLD'}
            Can either be a a single int (in which case the same activation is applied to all layers) or 
            a list of same length and shape as `conv_channels`.
            
            conv_kernel (optional): The size of the 3D convolutional kernels to be used. 
            Can either be a list of lists of same lengths as `conv_channels` or a single int. In the
             former case each value in the list represents the kernel size of that particular
            layer and in the latter case all the layers are built with the same kernel size as 
            specified.

            conv_padding (optional): The amount of zero-paddings to be done along each dimension.
            Format same as conv_kernel.

            conv_stride (optional): The stride of the 3D convolutions.
            Format same as conv_kernel.

            deconv_out_padding (optional): The additional zero-paddings to be done to the output 
            of ConvTranspose / Deconvolutions in the decoder network.
            By default does (stride-1) number of padding.
            Format same as conv_kernel.
            
            pool_type (optional): The type of pooling to be used. Options are (1)"max"  (2)"avg" 
            
            pool_kernel, pool_padding, pool_stride (optional): Can either be a single int or a list
            of respective pooling parameter values.
            The length of these list must be same as length of conv_channels i.e. the number of layers. 
            
            second_fc_decoder (optional): By default this is disabled. 
            If a non-empty list of ints is provided then a secondary decoder of a fully-connected network  
            is constructed as per the list.
        '''
        
        super().__init__(conv_channels)
        
        assert (self.is_nested_conv), "The conv_channels must be a list of list of ints Ex. [[16],[32 64],[64],...] (i.e. number of channels).\
It cannot be a list."
        
        self.conv_kernel = self.assign_parameter(conv_kernel, "conv_kernel")
        self.conv_padding = self.assign_parameter(conv_padding, "conv_padding")
        self.conv_stride = self.assign_parameter(conv_stride, "conv_stride")
        self.pool_kernel = self.assign_parameter(pool_kernel, "pool_kernel", enable_nested=False)
        self.pool_padding = self.assign_parameter(pool_padding, "pool_padding", enable_nested=False)
        self.pool_stride = self.assign_parameter(pool_stride, "pool_stride", enable_nested=False)        
        
        self.activation_fn = self.assign_parameter(activation_fn, "activation_function")

        for activations in self.activation_fn:
            for activation in activations:
                assert activation.upper() in self.valid_activations.keys(), "activation functions can only be one of the following str :\n {}".format(
                    self.valid_activations.keys())
        
        self.deconv_channels = self.nested_reverse(self.conv_channels)
        self.deconv_kernel = self.nested_reverse(self.conv_kernel)
        self.deconv_padding = self.nested_reverse(self.conv_padding)
        self.deconv_stride = self.nested_reverse(self.conv_stride)
        
        # set the switches used by forward() as false by default
        self.debug = False
        self.return_encoder_out = False
        
        if(deconv_out_padding is not None):
            self.deconv_out_padding = self.nested_reverse(
                self.assign_parameter(deconv_out_padding, "deconv_out_padding")
            )                
        else:
            self.deconv_out_padding = [[s-1 for s in layer] for layer in self.deconv_stride]
            
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.deconvs = nn.ModuleList()
        self.unpools = nn.ModuleList()
        

        for i in range(self.layers):
            
            self.convs.append(
                nn.ModuleList(
                    [self.add_conv_with_activation(
                        inner_conv_channels[0], inner_conv_channels[1]
                        , self.conv_kernel[i][j]
                        , self.conv_padding[i][j]
                        , self.conv_stride[i][j]
                        , self.activation_fn[i][j]) \
                    for j, inner_conv_channels in enumerate(self.conv_channels[i])]
                    )
                )

            self.deconvs.append(
                nn.ModuleList(
                    [self.add_deconv_with_activation(
                        inner_deconv_channels[0], inner_deconv_channels[1] 
                        , self.deconv_kernel[i][j]
                        , self.deconv_padding[i][j]
                        , self.deconv_stride[i][j]
                        , self.deconv_out_padding[i][j]
                        , self.activation_fn[i][j]) \
                    for j, inner_deconv_channels in enumerate(self.deconv_channels[i])]
                    )
                )

            self.pools.append(
                self.add_pool(
                    pool_type,
                    self.pool_kernel[i], 
                    stride = self.pool_stride[i], 
                    padding = self.pool_padding[i]
                )
            )
            self.unpools.append(
                self.add_unpool(
                    pool_type,
                    self.pool_kernel[-i-1], 
                    stride = self.pool_stride[-i-1], 
                    padding = self.pool_padding[-i-1]
                )
            )
        
    def set_debug(self, bool_val):
        self.debug = bool_val
    
    def set_return_encoder_out(self, bool_val):
        self.return_encoder_out = bool_val
        
    def forward(self, x):
            '''return_encoder_out : If enabled returns a list with 2 values, 
            first one is the Autoencoder's output and the other the intermediary output of the encoder.
            '''
            pool_idxs = []
            pool_sizes = [x.size()] #https://github.com/pytorch/pytorch/issues/580
            
            if(self.debug):
                print("\nImage dims ="+str(x.size()))
                
            #encoder
            for i,(convs, pool) in enumerate(zip(self.convs, self.pools)):
                for j, conv in enumerate(convs):
                    x = conv(x)
                    if(self.debug):print("conv{}{} output dim = {}".format(i+1, j+1, x.size()))
                        
                x, idx = pool(x)
                pool_sizes.append(x.size()) 
                pool_idxs.append(idx)
                if(self.debug):print("pool{} output dim = {}".format(i+1, x.size()))
            
            encoder_out = x
            
            if(self.debug):
                print("\nEncoder output dims ="+str(encoder_out.size())+"\n")
                
            #decoder
            pool_sizes.pop() # pop out the last size as it is not necessary

            for i,(deconvs, unpool) in enumerate(zip(self.deconvs, self.unpools)):

                x = unpool(x, pool_idxs.pop(), output_size=pool_sizes.pop())
                if(self.debug):print("unpool{} output dim = {}".format(i+1, x.size()))
                
                for j, deconv in enumerate(deconvs):
                    x = deconv(x)
                    if(self.debug):print("deconv{}{} output dim = {}".format(i+1, j+1, x.size()))
                        
            if(self.debug):
                print("\nDecoder output dims ="+str(x.size())+"\n")
                
            if(self.return_encoder_out):
                return [x, encoder_out]
            else:
                return x
            
            
            
            
class MLP(nn.Module):
    '''
    Constructs fully-connected deep neural networks 
    '''
    def __init__(self
        , layers = []
        , output_activation = nn.LogSoftmax
        ):
        '''
        Args:
            layer_neurons : Each value represents the number of neurons in each layer. The length of the list
            defines the number of layers. '''
        super().__init__()   
        self.layers = self._format_channels(layers)        
#         self.output_activation = output_activation
        self.debug = False
        
        # build the fully-connected layers
        self.fcs = nn.ModuleList()

        for layer in self.layers:
            if(layer) is not self.layers[-1]:
                self.fcs.append(self.add_linear_with_Relu(layer))
            elif(output_activation is not None):
                self.fcs.append(
                    nn.Sequential(
                        nn.Linear(layer[0], layer[1]),
                        output_activation()))
            else:
                self.fcs.append(
                    nn.Linear(layer[0], layer[1]))
                
    def set_debug(self, bool_val):
        self.debug = bool_val
        
    def _format_channels(self, layers):   
        layer_inout = []
        for i in range(len(layers)-1):
            layer_inout.append([layers[i], layers[i+1]])
        return layer_inout

    def add_linear_with_Relu(self, layer):
        node = nn.Sequential(
            nn.Linear(layer[0], layer[1]),
            nn.ReLU(True))        
        return node
        
    def forward(self, x):
        for i,fc in enumerate(self.fcs):
            x = fc(x)
            if(self.debug):print("FC {} output dims ={}".format(i, x.size()))
                
        return x