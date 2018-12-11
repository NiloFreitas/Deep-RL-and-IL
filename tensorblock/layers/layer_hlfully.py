
import numpy as np
import tensorflow as tf
import tensorblock as tb

class layer_hlfully:

####### Data

    def name(): return 'Fully'
    def shapeMult(): return 1
    def dims(): return 1

    def allowPooling(): return False

####### Function
    def function( x , W , b , recipe , pars ):

        if tb.aux.tf_length( x ) > 2:
            x = tb.aux.tf_flatten( x )

        layer = tf.contrib.layers.fully_connected (x,
                                                   pars['out_channels'],
                                                   activation_fn = None,
                                                   weights_regularizer = tf.contrib.layers.l2_regularizer(0.1))

        return [ layer ] , pars , None

####### Shapes
    def shapes( input_shape , pars ):

        in_channels = tb.aux.flat_dim( input_shape )
        out_channels =  pars['out_channels'] * np.prod( pars['out_sides'] )

        weight_shape = [ in_channels , out_channels ]
        bias_shape   = [               out_channels ]

        return weight_shape , bias_shape
