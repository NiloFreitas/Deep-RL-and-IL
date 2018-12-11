
import tensorflow as tf
import tensorblock as tb

class recipe_base:

####### Get Collection
    def collection( self , name = None ):

        if name is None: name = self.folder
        return tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , name )

####### Get Tag
    def tag( self , name ):

        list = name.split( '/' )

        if len( list ) == 1 :
            return self.labels[ name ][0]
        else:
            ptr = self.iterate( list )
            return ptr.tag( list[-1] )

####### Get Extra
    def info( self , name ):
        return self.data( name , get_info = True )

####### Get Data
    def data( self , name , get_info = False ):

        if not isinstance( name , str ):
            return [ name , None ]

        list = name.split( '/' )

        if len( list ) == 1 :

            tag , idx = self.labels[ name ]

            if tag == 'input':     return self.inputs[     idx ]
            if tag == 'weight':    return self.weights[    idx ]
            if tag == 'bias':      return self.biases[     idx ]
            if tag == 'variable':  return self.variables[  idx ]
            if tag == 'operation': return self.operations[ idx ]
            if tag == 'block':     return self.blocks[     idx ]
            if tag == 'summary':   return self.summaries[  idx ]
            if tag == 'writer':    return self.writers[    idx ]
            if tag == 'saver':     return self.savers[     idx ]
            if tag == 'plotter':   return self.plotters[   idx ]

            if tag == 'layer':
                if get_info: return self.extras[ idx ]
                else: return self.layers[ idx ]

            if tag == 'dropout':
                return self.root.dropouts[ idx ]

            return None

        else:

            ptr = self.iterate( list )
            return ptr.data( list[-1] )

####### Get Parameters
    def pars( self , name ):
        return self.data( name )[1]

####### Get Shape
    def shape( self , name ):
        node = self.node( name )
        if isinstance( node , list ): node = node[0]
        return tb.aux.tf_shape( node )

####### Get Tensor
    def node( self , name ):
        return self.data( name )[0]

####### Get Tensor
    def tensor( self , names ):

        tensors = self.tensor_list( names )
        if not isinstance( names , list ) and names[-1] != '/': return tensors[0]
        else: return tensors

####### Get Tensor List
    def tensor_list( self , names ):

        if not isinstance( names , list ):
            tensors = [ self.tensor_expanded( names ) ]
        else:
            tensors = []
            for name in names:
                tensors.append( self.tensor_expanded( name ) )
        return tb.aux.flatten_list( tensors )

####### Get Block Tensors
    def tensors_block( self ):

        tensors = []
        for label in self.labels:
            if self.tag( label ) == 'weight' and self.pars( label )['weight_type'] is not None:
                tensors.append( self.node( label ) )
            if self.tag( label ) == 'bias'   and self.pars( label )['bias_type'  ] is not None:
                tensors.append( self.node( label ) )
            if self.tag( label ) == 'block':
                tensors.append( self.block( label ).tensors_block() )
        return tb.aux.flatten_list( tensors )

####### Get Tensor Expanded
    def tensor_expanded( self , name ):

        if not isinstance( name , str ):
            return name

        expand = name[-1] == '/'
        if not expand: return self.node( name )
        else: name = name[:-1]

        if self.tag( name ) == 'layer':

            pars = self.pars( name ).copy()

            tensors = []
            if pars['weight_type'] is not None:
                tensors.append( self.node( pars['folder'] + pars['weight_name'] ) )
            if pars['bias_type'] is not None:
                tensors.append( self.node( pars['folder'] + pars['bias_name'  ] ) )
            return tensors

        elif self.tag( name ) == 'block':

            return self.block( name ).tensors_block()

        else:

            return [ self.node( name ) ]

####### Get Block
    def block( self , name ):

        list = name.split( '/' )
        if len( list ) == 1 : return self.tensor( name )
        else: return self.iterate( list )[ list[-1] ]

####### Get Block
    def __getitem__( self , name ):
        return self.block( name )

####### Iterate
    def iterate( self , list ):

        if list[0] is '':
            ptr = self.root
        else:
            if list[0][0] is '.':
                ptr = self.root
                back = self.folder.split( '/' )
                for i in range( len( back ) - len( list[0] ) ):
                    if back[i] is not '' : ptr = ptr[ back[i] ]
                list = list[1:]
            else:
                ptr = self

        for i in range( len( list ) - 1 ):
            if list[i] is not '' : ptr = ptr[ list[i] ]

        return ptr

####### Add Label
    def add_label( self , list , string , name , add_order ):

        idx = len( list )
        if name is None: name = string + '_' + str( idx )
        self.labels[ name ] = ( string.lower() , idx )

        if add_order:
            self.order.append( [ string.lower() , name ] )

        return name
