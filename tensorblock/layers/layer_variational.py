
import numpy as np
import tensorflow as tf
import tensorblock as tb

class layer_variational:

####### Data

    def name(): return 'Variational'
    def shapeMult(): return 2
    def dims(): return 1

    def allowPooling(): return False

####### Function
    def function( x , W , b , recipe , pars ):

        if tb.aux.tf_length( x ) > 2:
            x = tb.aux.tf_flatten( x )

        in_channels = pars['in_channels']
        out_channels = pars['out_channels']
        size_batch = tf.shape( x , name = 'batch' )[0]

        layer = tf.matmul( x , W , name = 'MatMul' )
        layer = tb.extras.bias( layer , b )

        z_mu = layer[ : , :out_channels ]
        z_sig = layer[ : , out_channels: ]

        out_shape = tf.stack( [ size_batch , out_channels ] , name = 'shape' )
        epsilon = tf.random_normal( out_shape , mean = 0.0 , stddev = 1.0 , seed = 1 )

        layer = tf.add( z_mu , tf.multiply( z_sig , epsilon ) )

        return [ layer , z_mu , z_sig ] , pars , None

####### Shapes
    def shapes( input_shape , pars ):

        in_channels = tb.aux.flat_dim( input_shape )
        out_channels =  pars['out_channels'] * np.prod( pars['out_sides'] )

        weight_shape = [ in_channels , out_channels ]
        bias_shape   = [               out_channels ]

        return weight_shape , bias_shape





