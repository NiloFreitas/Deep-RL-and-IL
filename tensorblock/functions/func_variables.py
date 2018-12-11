
import tensorflow as tf

### Placeholder
def placeholder( shape = None , name = None , dtype = tf.float32 ):

    if shape is None: return tf.placeholder( dtype , name = name )
    return tf.placeholder( dtype , shape , name = name )

### Numpy
def numpy( tensor , pars , name = None , dtype = tf.float32 ):

    return tf.Variable( tensor , trainable = pars['trainable'] , dtype = dtype )

### Copy
def copy( tensor , pars , name = None , dtype = tf.float32 ):

    return tf.Variable( tensor.initialized_value() , trainable = pars['trainable'] )

### Dummy
def dummy( shape , name = None , dtype = tf.float32 ):

    if name is None: name = 'dummy'
    return tf.get_variable( initializer = tf.zeros( shape = shape ) ,
                trainable = False , dtype = dtype , name = name )

### None
def none( shape , pars , name = None , dtype = tf.float32 ):

    return None

### Constant
def constant( shape , pars , name = None , dtype = tf.float32 ):

    if name is None: name = 'constant'
    return tf.get_variable( initializer = tf.constant( pars['value'] , shape = shape ) ,
                trainable = pars['trainable'] , dtype = dtype , name = name )

### Random Normal
def random_normal( shape , pars , name = None , dtype = tf.float32 ):

    if name is None: name = 'random_normal'
    return tf.get_variable( initializer = tf.random_normal( shape = shape ,
                mean = pars['mean'] , stddev = pars['stddev'] , seed = pars['seed'] ) ,
                trainable = pars['trainable'] , dtype = dtype , name = name )

### Truncated Normal
def truncated_normal( shape , pars , name = None , dtype = tf.float32 ):

    if name is None: name = 'truncated_normal'
    return tf.get_variable( initializer = tf.truncated_normal( shape = shape ,
                mean = pars['mean'] , stddev = pars['stddev'] , seed = pars['seed'] ) ,
                trainable = pars['trainable'] , dtype = dtype , name = name )

### Random Uniform
def random_uniform( shape , pars , name = None , dtype = tf.float32 ):

    if name is None: name = 'random_uniform'
    return tf.get_variable( initializer = tf.random_uniform( shape = shape ,
                minval = pars['min'] , maxval = pars['max'] , seed = pars['seed'] ) ,
                trainable = pars['trainable'] , dtype = dtype , name = name )

