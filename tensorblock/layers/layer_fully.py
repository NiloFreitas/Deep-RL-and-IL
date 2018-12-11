
import numpy as np
import tensorflow as tf
import tensorblock as tb

class layer_fully:

####### Data

    def name(): return 'Fully'
    def shapeMult(): return 1
    def dims(): return 1

    def allowPooling(): return False

####### Function
    def function( x , W , b , recipe , pars ):

        if tb.aux.tf_length( x ) > 2:
            x = tb.aux.tf_flatten( x )

        layer = tf.matmul( x , W , name = 'MatMul' )
        layer = tb.extras.bias( layer , b )

        return [ layer ] , pars , None

####### Shapes
    def shapes( input_shape , pars ):

        in_channels = tb.aux.flat_dim( input_shape )
        out_channels =  pars['out_channels'] * np.prod( pars['out_sides'] )

        weight_shape = [ in_channels , out_channels ]
        bias_shape   = [               out_channels ]

        return weight_shape , bias_shape
