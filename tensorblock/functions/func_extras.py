
import tensorflow as tf
import tensorblock as tb

### 2D Max Pooling
def maxpool2d( x , pars ):

    if not pars['pooling_ksize'  ]: pars['pooling_ksize']   = pars['pooling']
    if not pars['pooling_strides']: pars['pooling_strides'] = pars['pooling']
    if not pars['pooling_padding']: pars['pooling_padding'] = pars['padding'    ]

    ksize   = tb.aux.spread( pars['pooling_ksize'  ] , 2 )
    strides = tb.aux.spread( pars['pooling_strides'] , 2 )

    return tf.nn.max_pool( x , ksize   = [ 1 , ksize[0]   , ksize[1]   , 1 ] ,
                               strides = [ 1 , strides[0] , strides[1] , 1 ] ,
                               padding = pars['pooling_padding'] )

### 2D Max Pooling
def maxpool3d( x , pars ):

    if not pars['pooling_ksize'  ]: pars['pooling_ksize']   = pars['pooling']
    if not pars['pooling_strides']: pars['pooling_strides'] = pars['pooling']
    if not pars['pooling_padding']: pars['pooling_padding'] = pars['padding']

    ksize   = tb.aux.spread( pars['pooling_ksize'  ] , 3 )
    strides = tb.aux.spread( pars['pooling_strides'] , 3 )

    return tf.nn.max_pool3d( x , ksize   = [ 1 , ksize[0]   , ksize[1]   , ksize[2]   , 1 ] ,
                                 strides = [ 1 , strides[0] , strides[1] , strides[2] , 1 ] ,
                                 padding = pars['pooling_padding'] )

### Bias
def bias( x , b ):

    if b != None:
        x = tf.nn.bias_add( x , b )
    return x

### Dropout
def dropout( x , dropout ):

    return tf.nn.dropout( x , dropout )

### Normal Distribution
def dist_normal( mu, sigma ):

    return tf.distributions.Normal( mu, sigma )

### Log Sigmoid
def log_sig( x ):

    return - tf.log( 1 - tf.nn.sigmoid( x ) + 1e-8 )
