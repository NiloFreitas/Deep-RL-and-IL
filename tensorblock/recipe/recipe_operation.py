
import tensorflow as tf

class recipe_operation:

####### Add Operation
    def addOperation( self , **args ):

        pars = { **self.defs_operation , **args }

        pars['name'] = self.add_label(
                self.operations , 'Operation' , pars['name'] , add_order = True )

        if pars['input'] is not None:
            tensors = self.tensor_list( pars['input'] )
        if pars['src'] is not None and pars['dst'] is not None:
            tensors = [ self.tensor_list( pars['src'] ) ,
                        self.tensor_list( pars['dst'] ) ]

        extras = pars['extra']
        if extras is not None:
            extras = self.info( extras )

        if callable( pars['function'] ):
            with tf.variable_scope( self.folder + pars['name'] ):
                self.operations.append( [ pars['function']( tensors , extras , pars ) , pars ] )
        else:
            self.operations.append( [ pars['function'] , pars ] )


        return self.operations[-1][0]
