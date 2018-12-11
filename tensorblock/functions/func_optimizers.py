
import tensorflow as tf

### Adam Optimizer
def adam( tensors , extras , pars ):

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(extra_update_ops):
        trainer = tf.train.AdamOptimizer( pars['learning_rate'] )
        update = trainer.minimize( tensors[0] )

    return update

### Gradient Descent Optimizer
def gradient_descent( tensors , extras , pars ):

    return tf.train.GradientDescentOptimizer( pars['learning_rate'] ).minimize( tensors[0] )

### Adam Optimizer
def adam_apply( tensors , extras , pars ):

    vars = tf.trainable_variables()
    normal_actor_vars = [var for var in vars if 'NormalActor' in var.name]

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(extra_update_ops):
        trainer = tf.train.AdamOptimizer( pars['learning_rate'] )
        update = trainer.apply_gradients(zip(tensors,  normal_actor_vars  ) )

    return update
