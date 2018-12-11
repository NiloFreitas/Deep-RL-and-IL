
import tensorflow as tf
import tensorblock as tb

class recipe_print:

####### Print Nodes
    def printNodes( self ):

        print( '############################################################## TENSORS' ,
                '/' + self.folder[0:-1] )

        print( '#################################################### INPUTS' )
        for input in self.inputs:

            pars = input[1]
            print( '*************************** ' + pars['name'] , end = '' )
            print( ' | S: ' + str( self.shape( pars['name'] ) )  , end = '' )
            print( ' |'                                                     )

        print( '#################################################### LAYERS' )
        for layer in self.layers:

            type , input , name = layer[1]['type'].name() , layer[1]['input'] , layer[1]['name']
            weight_shape , bias_shape = self.shape( layer[1]['weight_name'] ) , self.shape( layer[1]['bias_name'] )
            dropout , pooling = layer[1]['dropout'] , layer[1]['pooling']
            in_dropout , out_dropout = layer[1]['in_dropout'] , layer[1]['out_dropout']
            W = self.node( layer[1]['weight_name'] )
            strW = ' W:' if not isinstance( W , list ) else ' W(' + str( len( W ) ) + '):'

            b = self.node( layer[1]['bias_name'] )
            strB = ' b:' if not isinstance( b , list ) else ' b(' + str( len( W ) ) + '):'

            print( '*************************** (' + type + ') - ' + input + ' --> ' + name , end = '' )
            print( ' | I:' , self.shape( layer[1]['input'] )            , end = '' )
            print( ' | O:' , tb.aux.tf_shape( layer[0] )                , end = '' )
            if weight_shape is not None : print( ' |' + strW    , weight_shape , end = '' )
            if bias_shape   is not None : print( ' |' + strB    , bias_shape   , end = '' )
            if pooling     >   1        : print( ' | pool:'     , pooling      , end = '' )
            if dropout     > 0.0        : print( ' | drop:'     , dropout      , end = '' )
            if in_dropout  > 0.0        : print( ' | in_drop:'  , in_dropout   , end = '' )
            if out_dropout > 0.0        : print( ' | out_drop:' , out_dropout  , end = '' )
            print( ' |' )

        print( '#################################################### VARIABLES' )
        for input in self.variables:

            pars = input[1]
            print( '*************************** ' + pars['name'] , end = '' )
            print( ' | S: ' + str( self.shape( pars['name'] ) )  , end = '' )
            print( ' |'                                                     )

        print( '#################################################### OPERATIONS' )
        for operation in self.operations:

            name = operation[1]['name']
            inputs , src , dst = operation[1]['input'] , operation[1]['src'] , operation[1]['dst']
            print( '*************************** ' , end = '' )
            if inputs is not None:
                if not isinstance( inputs , list ):
                    print( inputs , end = '' )
                else:
                    for i in range( len(inputs) - 1 ):
                        print( inputs[i] + ' & ' , end = '' )
                    print( inputs[-1] , end = '' )
            if src is not None:
                if not isinstance( src , list ):
                    print( src + ' <-> ' + dst , end = '' )
                else:
                    for i in range( len(src) - 1 ):
                        print( src[i] + ' & ' , end = '' )
                    print( src[-1] , end = '' )
                    print( ' <-> ' , end = '' )
                    for i in range( len(dst) - 1 ):
                        print( dst[i] + ' & ' , end = '' )
                    print( dst[-1] , end = '' )
            print( ' --> ' + name )

        print( '############################################################## END TENSORS' ,
                '/' + self.folder[0:-1] )

####### Print All Nodes
    def printAllNodes( self ):

        self.printNodes()
        for i in range( len( self.blocks ) ):
            self.blocks[i][0].printAllNodes()

####### Print Collection
    def printCollection( self ):

        print( '############################################################## COLLECTION' )

        vars = [ v for v in tf.global_variables() if v.name.startswith( self.folder ) ]
        for var in vars: print( var.name )

        print( '############################################################## END COLLECTION' )

####### Print All Collection
    def printAllCollection( self ):

        print( '############################################################## COLLECTION' )

        vars = [ v for v in tf.global_variables() ]
        for var in vars: print( var.name )

        print( '############################################################## END COLLECTION' )

