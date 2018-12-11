
import numpy as np
import tensorflow as tf
import tensorblock as tb

class layer_deconv2d:

####### Data

    def name(): return 'Deconv2D'
    def shapeMult(): return 1
    def dims(): return 2

    def allowPooling(): return False

####### Function
    def function( x , W , b , recipe , pars ):

        if tb.aux.tf_length( x ) == 2:
            x = tb.aux.tf_fold2D( x , tb.aux.tf_shape( W )[-1] )
            pars['in_sides'] = tb.aux.tf_shape( x )[1:3]

        out_channels = tb.aux.tf_shape( W )[-2]
        size_batch = tf.shape( x , name = 'batch' )[0]

        in_sides , out_sides , strides = pars['in_sides'] , pars['out_sides'] , pars['strides']

        if np.prod( out_sides ) == 1:
            for i in range( len( out_sides ) ):
                out_sides[i] = in_sides[i] * strides[i]

        strides = [ int( np.ceil( out_sides[0] / in_sides[0] ) ) ,
                    int( np.ceil( out_sides[1] / in_sides[1] ) ) ]

        out_shape = tf.stack( [ size_batch , out_sides[0] ,
                                             out_sides[1] , out_channels ] , name = 'shape' )

        layer = tf.nn.conv2d_transpose( x , W , name = 'Deconv2D' ,
                                        output_shape = out_shape ,
                                        strides = [ 1 , strides[0] , strides[1] , 1 ] ,
                                        padding = pars['padding'] )

        dummy = tb.vars.dummy( [ out_sides[0] , out_sides[1] , out_channels ] , name = 'dummy' )
        layer = tf.add( layer , dummy , name = 'DummyAdd' )

        layer = tb.extras.bias( layer , b )

        return [ layer ] , pars , None

####### Shapes
    def shapes( input_shape , pars ):

        in_channels = pars['in_channels']
        out_channels =  pars['out_channels']
        ksize = pars['ksize'];

        weight_shape = [ ksize[0] , ksize[1] , out_channels , in_channels ]
        bias_shape   = [                       out_channels ]

        return weight_shape , bias_shape

