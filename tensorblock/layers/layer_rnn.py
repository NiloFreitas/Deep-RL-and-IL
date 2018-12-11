
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorblock as tb

class layer_rnn:

####### Data

    def name(): return 'RNN'
    def shapeMult(): return 1
    def dims(): return 1

    def allowPooling(): return False


####### Function
    def function( x , W , b , recipe , pars ):

        # Fold Data if Necessary
        if tb.aux.tf_length( x ) == 2:

            in_sides = pars['in_sides']
            if len( in_sides ) == 1: x = tb.aux.tf_fold2D( x , 1 )
            else: x = tb.aux.tf_fold( x , in_sides )

        # Prepare Network
        with tf.variable_scope( 'RNN' ):

            # Create Cell

            if pars['cell_type'] == 'LSTM': # LSTM
                cell = rnn.BasicLSTMCell( pars['out_channels'] , forget_bias = 1.0 , state_is_tuple = True )
            else: # GRU
                cell = rnn.GRUCell( pars['out_channels'] )

            # Input Dropout
            if pars['in_dropout'] > 0.0 :

                in_name = pars['in_dropout_name']
                if in_name is None : in_name = 'indrop_' + pars['name']

                idx = len( recipe.root.dropouts ) ; recipe.labels[ in_name ] = ( 'dropout' , idx )
                recipe.root.dropouts.append( [ tb.vars.placeholder( name = 'drop_Input' ) , [ idx , pars['in_dropout'] ] ] )
                cell = rnn.DropoutWrapper( cell , input_keep_prob = recipe.root.dropouts[-1][0] )

            # Output Dropout
            if pars['out_dropout'] > 0.0 :

                out_name = pars['out_dropout_name']
                if out_name is None : out_name = 'outdrop_' + pars['name']

                idx = len( recipe.root.dropouts ) ;  recipe.labels[ out_name ] = ( 'dropout' , idx )
                recipe.root.dropouts.append( [ tb.vars.placeholder( name = 'drop_Output' ) , [ idx , pars['out_dropout'] ] ] )
                cell = rnn.DropoutWrapper( cell , output_keep_prob = recipe.root.dropouts[-1][0] )

            # Stack Cells
            if pars['num_cells'] is not None:
                def lstm_cell(): return rnn.BasicLSTMCell( pars['out_channels'] , forget_bias = 1.0 , state_is_tuple = True )
                cell = rnn.MultiRNNCell( [ lstm_cell() for _ in range(pars['num_cells'])], state_is_tuple = True )

            # Create RNN
            outputs , states = tf.nn.dynamic_rnn( cell , x , dtype = tf.float32 , sequence_length = pars['seqlen'] )

        # Check Sequence Length
        if pars['seqlen'] is None: # Without Sequence Length

            shape = tb.aux.tf_shape( outputs )
            trans = list( range( len( shape ) ) )
            trans[0] , trans[1] = trans[1] , trans[0]

            lasts = tf.transpose( outputs , trans )[-1]

        else: # With Sequence Length

            with tf.variable_scope( 'Gather' ):

                batch_shape , batch_size = tb.aux.tf_shape( x ) , tf.shape( outputs )[0]
                index = tf.range( 0 , batch_size ) * batch_shape[1] + ( pars['seqlen'] - 1 )
                lasts = tf.gather( tf.reshape( outputs , [ -1 , pars['out_channels'] ] ) , index )

        # Store Weights and Biases
        if pars['num_cells'] is None: # Single Cell

            if pars['cell_type'] == 'LSTM':

                path = pars['folder'] + pars['name'] + '/RNN/rnn/basic_lstm_cell/'
                WW = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , path + 'kernel:0' )[0]
                bb = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , path + 'bias:0'  )[0]

            elif pars['cell_type'] == 'GRU':

                path = pars['folder'] + pars['name'] + '/RNN/rnn/gru_cell/'
                WW , bb = [] , []

                WW.append( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , path + 'gates/kernel:0'     )[0] )
                bb.append( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , path + 'gates/bias:0'      )[0] )
                WW.append( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , path + 'candidate/kernel:0' )[0] )
                bb.append( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , path + 'candidate/bias:0'  )[0] )

        else: # Stacked Cells

            path = pars['folder'] + pars['name'] + '/RNN/rnn/multi_rnn_cell/cell_'
            WW , bb = [] , []

            if pars['cell_type'] == 'LSTM':

                for i in range( pars['num_cells'] ):
                    pathi = path + str( i ) + '/basic_lstm_cell/'
                    path = pars['folder'] + pars['name'] + '/RNN/rnn/basic_lstm_cell/'
                    WW.append( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , pathi + 'kernel:0' ) )
                    bb.append( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , pathi + 'bias:0'  ) )

            elif pars['cell_type'] == 'GRU':

                for i in range( pars['num_cells'] ):
                    pathi = path + str( i ) + '/gru_cell/'
                    path = pars['folder'] + pars['name'] + '/RNN/rnn/gru_cell/'
                    WW.append( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , pathi + 'gates/kernel:0'     ) )
                    bb.append( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , pathi + 'gates/bias:0'      ) )
                    WW.append( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , pathi + 'candidate/kernel:0' ) )
                    bb.append( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , pathi + 'candidate/bias:0'  ) )


        # Return Layer
        return [ lasts , outputs , states ] , pars , [ WW , bb ]

####### Shapes
    def shapes( input_shape , pars ):

        return None , None
