
import numpy as np
import tensorflow as tf
import tensorblock as tb

class layer_conv1d:

####### Data

    def name(): return 'Conv1D'
    def shapeMult(): return 1
    def dims(): return 1

    def allowPooling(): return True

####### Function
    def function( x , W , b , recipe , pars ):

        strides = pars['strides']
        layer = tf.nn.conv1d( x , W , name = 'Conv1D' ,
                                      stride = strides[0] ,
                                      padding = pars['padding'] )
        layer = tb.extras.bias( layer , b )

        return [ layer ] , pars , None

####### Shapes
    def shapes( input_shape , pars ):

        in_channels = pars['in_channels']
        out_channels =  pars['out_channels']
        ksize = pars['ksize']

        weight_shape = [ ksize[0] , in_channels , out_channels ]
        bias_shape   = [                          out_channels ]

        return weight_shape , bias_shape
