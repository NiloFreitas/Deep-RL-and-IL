
import tensorblock as tb

class layer_flatten:

####### Data

    def name(): return 'Flatten'
    def shapeMult(): return None
    def dims(): return 1

    def allowPooling(): return False

####### Function
    def function( x , W , b , recipe , pars ):

        layer = tb.aux.tf_flatten( x )
        return [ layer ] , pars , None

####### Shapes
    def shapes( input_shape , pars ):

        return None , None
