#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorblock.aux.aux_reshape import *

### Get Shape
def tf_shape( x ):

    if x is None: return None
    return x.get_shape().as_list()

### Get Length
def tf_length( x ):

    return len( tf_shape( x ) )

### Flat Dimension
def tf_flat_dim( x ):

    return flat_dim( tf_shape( x ) )

### Flatten
def tf_flatten( x ):

    return tf.reshape( x , [ -1 , tf_flat_dim( x ) ] )

### 2D Side Dimension
def tf_side2D( x , d = 1 ):

    return side2D( tf_shape( x ) , d )

### 3D Side Dimension
def tf_side3D( x , d = 1 ):

    return side3D( tf_shape( x ) , d )

### Fold
def tf_fold( x , dims ):

    return tf.reshape( x , [ -1 ] + dims )

### 2D Fold
def tf_fold2D( x , d = 1 ):

    s = tf_side2D( x , d )
    return tf.reshape( x , shape = [ -1 , s , s , d ] )

### 3D Fold
def tf_fold3D( x , d = 1 ):

    s = tf_side3D( x , d )
    return tf.reshape( x , shape = [ -1 , s , s , s , d ] )

### Squeeze
def tf_squeeze( x , d ):

    return tf.squeeze( x, d )

## Concat
def tf_concat( x , y, d ):

    return tf.concat( [x, y], d )
