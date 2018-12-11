
import tensorflow as tf
import tensorblock as tb
import numpy as np

class recipe_input:

####### Add Input
    def addInput( self , **args ):

        pars = { **self.defs_input , **args }
        pars['name'] = self.add_label(
                self.inputs , 'Input' , pars['name'] , add_order = True )
        pars = self.parse_input_pars( pars )

        if pars['share'] is not None:
            self.inputs.append( [ self.node( pars['share'] ) , pars ] )
        else:
            if pars['tensor'] is None:
                with tf.variable_scope( self.folder + pars['name'] , reuse = False ):
                    self.inputs.append( [ tb.vars.placeholder( shape = pars['shape'] ,
                                                               dtype = pars['dtype'] ) , pars ] )
            else: self.inputs.append( [ pars['tensor'] , pars ] )

        self.curr_input = pars['name']
        return self.inputs[-1][0]

####### Add Variable
    def addVariable( self , **args ):

        pars = { **self.defs_variable , **args }
        pars['name'] = self.add_label(
                self.variables , 'Variable' , pars['name'] , add_order = True )
        pars = self.parse_input_pars( pars )

        if pars['share'] is not None:
            self.variables.append( [ self.node( pars['share'] ) , pars ] )
        else:
            if pars['tensor'] is None:
                with tf.variable_scope( self.folder + pars['name'] , reuse = False ):
                    self.variables.append( [ pars['type']( pars['shape'] , pars ) , pars ] )
            else:
                if callable( pars['tensor'] ):
                    with tf.variable_scope( self.folder + pars['name'] , reuse = False ):
                        self.variables.append( [ pars['tensor']( pars['shape'] , pars ) , pars ] )
                else:
                    if isinstance( pars['tensor'] , np.ndarray ):
                        self.variables.append( [ tb.vars.numpy( pars['tensor'] , pars ) , pars ] )
                    else:
                        self.variables.append( [ pars['tensor'] , pars ] )

        return self.variables[-1][0]

####### Parse Pars
    def parse_input_pars( self , pars ):

        if pars['tensor'] is not None:

            pars['first_none'] = False

            if isinstance( pars['tensor'] , np.ndarray ):
                pars['shape'] = pars['tensor'].shape
            else:
                pars['shape'] = tb.aux.tf_shape( pars['tensor'] )

        if pars['copy'] is not None: # Copying

            pars['type'] = tb.vars.copy
            pars['shape'] = self.node( pars['copy'] )

            copy_pars = self.pars( pars['copy'] )
            pars['out_sides'] = copy_pars['out_sides']
            pars['out_channels'] = copy_pars['out_channels']

        else: # Nothing

            pars['shape'] = list( pars['shape'] )
            if pars['first_none'] and len( pars['shape'] ) > 1: pars['shape'][0] = None
            shape = pars['shape']

            if pars['out_sides'] is None:
                if len( shape ) == 2: pars['out_sides'] = shape[1:2] ;
                if len( shape ) == 4: pars['out_sides'] = shape[1:3] ;
                if len( shape ) == 5: pars['out_sides'] = shape[1:4] ;

            if pars['out_channels'] is None:
                if len( shape ) == 2: pars['out_channels'] = 1
                else: pars['out_channels'] = shape[-1]

        return pars
