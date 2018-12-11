
import numpy as np
import tensorflow as tf
import tensorblock as tb

class recipe_layer:

####### Add Layer
    def addLayer( self , **args ):

        pars = { **self.defs_layer , **args }
        if pars['input'] is None: pars['input'] = self.curr_input

        pars = self.parsePars( pars )
        pars = self.createLayerNames( pars )

        layer = pars['type']
        input_shape = self.shape( pars['input'] )
        pars['folder'] = self.folder

        weight_shape , bias_shape = layer.shapes( input_shape , pars )

        self.newLayerWeight( weight_shape , pars )
        self.newLayerBias(   bias_shape   , pars )
        self.newLayer(                      pars )

        self.curr_input = pars['name']

        return self.layers[-1][0]

####### Parse Pars
    def parsePars( self , pars ):

        if pars['copy'] is not None:
            folder = tb.aux.get_folder( pars['copy'] )
            copy_pars = self.pars( pars['copy'] )
            pars['out_channels'] =          copy_pars['out_channels']
            pars['type'        ] =          copy_pars['type'        ]
            pars['weight_copy' ] = folder + copy_pars['weight_name' ]
            pars['bias_copy'   ] = folder + copy_pars['bias_name'   ]
            pars['dropout_copy'] = folder + copy_pars['dropout_name']
            pars['dropout'     ] =          copy_pars['dropout' ]

        if pars['share'] is not None:
            folder = tb.aux.get_folder( pars['share'] )
            share_pars = self.pars( pars['share'] )
            pars['out_channels' ] =          share_pars['out_channels']
            pars['type'         ] =          share_pars['type'        ]
            pars['weight_share' ] = folder + share_pars['weight_name' ]
            pars['bias_share'   ] = folder + share_pars['bias_name'   ]
            pars['dropout_share'] = folder + share_pars['dropout_name']
            pars['dropout'      ] =          share_pars['dropout'     ]

        dims = pars['type'].dims()
        input_pars = self.pars( pars['input'] )

        if pars['in_sides'   ] is None: pars['in_sides'   ] = input_pars['out_sides'   ]
        if pars['in_channels'] is None: pars['in_channels'] = input_pars['out_channels']
        if pars['out_sides'  ] is None: pars['out_sides'  ] = 1

        if isinstance( pars['in_sides'] , str ):
            pars['in_sides'] = self.pars( pars['in_sides'] )['out_sides']
        if isinstance( pars['in_channels'] , str ):
            pars['in_channels'] = self.pars( pars['in_channels'] )['out_channels']

        if isinstance( pars['out_channels'] , str ):
            channels_pars = self.pars( pars['out_channels'] )
            if dims == 1 and len( channels_pars['out_sides'] ) == 1 :
                pars['out_channels'] = channels_pars['out_sides'][0]
            else: pars['out_channels'] = channels_pars['out_channels']

        if isinstance( pars['out_sides'] , str ):
            pars['out_sides'] = self.pars( pars['out_sides'] )['in_sides']
        if isinstance( pars['strides'] , str ):
            pars['strides'] = self.pars( pars['strides'] )['strides']

        if isinstance( pars['seqlen'] , str ):
            pars['seqlen'] = self.tensor( pars['seqlen'] )

        pars['ksize'    ] = tb.aux.spread( pars['ksize'    ] , dims )
        pars['strides'  ] = tb.aux.spread( pars['strides'  ] , dims )
        pars['in_sides' ] = tb.aux.spread( pars['in_sides' ] , dims )
        pars['out_sides'] = tb.aux.spread( pars['out_sides'] , dims )

        return pars

####### New Layer
    def newLayer( self , pars ):

        scope = self.folder + pars['name']
        with tf.variable_scope( scope , reuse = False ):

            layer , pars , vars = \
                pars['type'].function( self.node( pars['input'      ] ) , \
                                       self.node( pars['weight_name'] ) , \
                                       self.node( pars['bias_name'  ] ) , self , pars )

            layer = self.checkActivation( layer , pars )
            layer = self.checkPooling(    layer , pars )
            layer = self.checkDropout(    layer , pars )

            shape = tb.aux.tf_shape( layer[0] )
            if len( shape ) == 4: pars['out_sides'] = shape[1:3]
            if len( shape ) == 5: pars['out_sides'] = shape[1:4]

            if vars is not None:
                self.weights[-1][0] = vars[0]
                self.biases[ -1][0] = vars[1]

        self.layers.append( [ layer[0] , pars ] )
        self.extras.append( layer[1:] )

        self.cnt += 1

####### Check for Activation Function
    def checkActivation( self , layer , pars ):

        if pars['activation'] is not None:

            if pars['activation_pars'] is None: layer[0] = pars['activation']( layer[0] )
            else: layer[0] = pars['activation']( layer[0] , pars['activation_pars'] )

        return layer

####### Check for Pooling
    def checkPooling( self , layer , pars ):

        if pars['type'].allowPooling():
            if np.prod( pars['pooling'] ) > 1:

                dims = pars['type'].dims()
                if dims == 2: layer[0] = tb.extras.maxpool2d( layer[0] , pars )
                if dims == 3: layer[0] = tb.extras.maxpool3d( layer[0] , pars )

        return layer

####### Check for Dropout
    def checkDropout( self , layer , pars ):

        if pars['dropout'] == 0.0:
            return layer

        scope = pars['dropout_name']
        with tf.variable_scope( scope , reuse = False ):

            if pars['dropout_share'] is not None: # Sharing

                idx_share = self.pars( pars['dropout_share'] )[0]
                self.labels[ pars['dropout_name'] ] = ( 'dropout' , idx_share )
                layer[0] = tb.extras.dropout( layer[0] , self.root.dropouts[idx_share][0] )

            else:

                if pars['dropout_copy'] is not None: # Copying
                    dropout = self.pars( pars['dropout_copy'] )[1]
                else: dropout = pars['dropout']

                idx = len( self.root.dropouts )
                self.labels[ pars['dropout_name'] ] = ( 'dropout' , idx )
                self.root.dropouts.append( [ tb.vars.placeholder() , [ idx , dropout ] ] )
                layer[0] = tb.extras.dropout( layer[0] , self.root.dropouts[-1][0] )

        return layer

####### Set Dropout
    def setDropout( self , name = None , value = 1.0 ):
        self.pars( name )[1] = value

####### Add Dropout
    def addDropout( self , name = None , value = 1.0 ):
        self.pars( name )[1] += value

####### New Variable
    def newLayerVariable( self , shape , pars , type ):

        if type == 'weight': s , list = 'weight_' , self.weights
        if type == 'bias':   s , list = 'bias_'   , self.biases

        if shape is None: list.append( [ None , pars ] ) ; return
        if pars[s + 'type'] is None: list.append( [ None , pars ] ) ; return
        if isinstance( pars[s + 'type'] , str ):
            list.append( [ self.tensor( pars[s + 'type'] ) , pars ] ) ; return

        scope_name = '/' + pars[s + 'name']

        dict = { 'label' : s , 'mean' : pars[s + 'mean'] , 'stddev' : pars[s + 'stddev'] ,
                 'value' : pars[s + 'value'] , 'min' : pars[s + 'min'] , 'max' : pars[s + 'max'] ,
                 'trainable' : pars[s + 'trainable'] , 'seed' : pars[s + 'seed'] }

        if pars[s + 'share'] is not None: # Sharing

            share_pars = self.pars( pars[s + 'share'] )
            name = '/' + share_pars['folder'] + share_pars[s + 'name']
            list.append( [ self.node( name ) , pars ] ) ; return

        if pars[s + 'copy'] is not None: # Copying

            pars[s + 'type'] = tb.vars.copy
            copy_pars = self.pars( pars[s + 'copy'] )

            name = '/' + copy_pars['folder'] + copy_pars[s + 'name']
            shape = self.node( name )

        else: # Nothing

            shape[-1] = shape[-1] * pars['type'].shapeMult()

        scope = self.folder + pars['name'] + scope_name
        if scope[0] is '/' : scope = scope[1:]

        if callable( pars[s + 'type'] ):
            with tf.variable_scope( scope , reuse = False ):
                list.append( [ pars[s + 'type']( shape , dict , name = type ) , pars ] )
        else:
            if isinstance( pars[s + 'type'] , np.ndarray ):
                list.append( [ tb.vars.numpy( pars[s + 'type'] , dict , name = type ) , pars ] )
#                print( 'NUMPY' , pars[s + 'type'].shape )
            else:
                list.append( [ pars[s + 'type'] , pars ] )
#                print( 'TENSOR' )

####### New Weight
    def newLayerWeight( self , shape , pars ):
        self.newLayerVariable( shape , pars , 'weight' )

####### New Bias
    def newLayerBias( self , shape , pars ):
        self.newLayerVariable( shape , pars , 'bias' )

####### Create Layer Names
    def createLayerNames( self , pars ):

        if not pars['name'        ]: pars['name'        ] = pars['type'].name() + '_' + str( self.cnt )
        if not pars['weight_name' ]: pars['weight_name' ] = 'W_'    + pars['name']
        if not pars['bias_name'   ]: pars['bias_name'   ] = 'b_'    + pars['name']
        if not pars['dropout_name']: pars['dropout_name'] = 'drop_' + pars['name']

        self.labels[ pars['name'       ] ] = ( 'layer'  , self.cnt )
        self.labels[ pars['weight_name'] ] = ( 'weight' , self.cnt )
        self.labels[ pars['bias_name'  ] ] = ( 'bias'   , self.cnt )
        self.order.append( [ 'layer' , pars['name'] ] )

        return pars
