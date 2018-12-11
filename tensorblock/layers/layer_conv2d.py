
import numpy as np
import tensorflow as tf
import tensorblock as tb

class layer_conv2d:

####### Data

    def name(): return 'Conv2D'
    def shapeMult(): return 1
    def dims(): return 2

    def allowPooling(): return True

####### Function
    def function( x , W , b , recipe , pars ):

        if tb.aux.tf_length( x ) == 2:
            x = tb.aux.tf_fold2D( x , tb.aux.tf_shape( W )[-2] )
            pars['in_sides'] = tb.aux.tf_shape( x )[1:3]

        strides = pars['strides']
        layer = tf.nn.conv2d( x , W , name = 'Conv2D' ,
                                      strides = [ 1 , strides[0] , strides[1] , 1 ] ,
                                      padding = pars['padding'] )
        layer = tb.extras.bias( layer , b )

        return [ layer ] , pars , None

####### Shapes
    def shapes( input_shape , pars ):

        in_channels = pars['in_channels']
        out_channels =  pars['out_channels']
        ksize = pars['ksize']

        weight_shape = [ ksize[0] , ksize[1] , in_channels , out_channels ]
        bias_shape   = [                                     out_channels ]

        return weight_shape , bias_shape
