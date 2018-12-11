
import tensorflow as tf

### ELU Activation

def elu( x ):

    return tf.nn.elu( x )

### ReLU Activation
def relu( x ):

    return tf.nn.relu( x )

### Tanh Activation
def tanh( x ):

    return tf.nn.tanh( x )

### SoftPlus Activation
def softplus( x, scale = 1):

    return tf.nn.softplus( x ) * scale

### SoftMax Activation
def softmax( x ):

    return tf.nn.softmax( x )

### Sigmoid Activation
def sigmoid( x ):

    return tf.nn.sigmoid( x )

### Squared Exponential Activation
def sqrexp( x , pars ):

    values = tf.Variable( pars , trainable = False , dtype = tf.float32 )
    return values[0] * tf.exp( - tf.square( x ) / ( 2.0 * values[1] ) )
