from players_reinforcement.player_reinforce_rnn_2 import *
import tensorflow as tf


# PLAYER REINFORCE RNN
class player_reinforce_rnn_2A( player_reinforce_rnn_2 ):

    NUM_FRAMES      = 3
    LEARNING_RATE   = 1e-4
    REWARD_DISCOUNT = 0.99

    ### __INIT__
    def __init__( self ):

        player_reinforce_rnn_2.__init__( self )

    # PREPARE NETWORK
    def network( self ):

        # Input Placeholder

        self.brain.addInput( shape = [ None, self.NUM_FRAMES, self.obsv_shape[0], self.obsv_shape[1] ],
                             name = 'Observation' )

        # Reshape Input to CNN (B,T,D1,D2)->(B*T,D1,D2,1)

        self.obs = self.brain.tensor('Observation')
        self.obs = tf.expand_dims ( self.obs, axis=tf.rank(self.obs) )
        self.obs = tf.reshape( self.obs, [ tf.shape(self.obs)[0]*self.NUM_FRAMES, self.obsv_shape[0], self.obsv_shape[1], 1 ] )
        self.brain.addInput( tensor = self.obs, name = 'InputCNN' )

        # Convolutional Layers

        self.brain.setLayerDefaults( type=tb.layers.conv2d,
                                     activation=tb.activs.relu, pooling=2, weight_stddev=0.01, bias_stddev=0.01 )

        self.brain.addLayer( out_channels=32, ksize=8, strides=4, input = 'InputCNN' )
        self.brain.addLayer( out_channels=64, ksize=4, strides=2 )
        self.brain.addLayer( out_channels=64, ksize=3, strides=1 )

        #  Fully

        self.brain.addLayer( type = tb.layers.flatten,
                             name = 'Flatten' )

        self.brain.addLayer( type = tb.layers.fully,
                             out_channels = 256 ,
                             activation   = tb.activs.elu,
                             name         = 'OutputFully' )

        # Reshape OutputFully to RNN (B*T,C)->(B,T,C)

        self.outfully = tf.reshape( self.brain.tensor('OutputFully') , [-1, self.NUM_FRAMES, 256] )

        self.brain.addInput( tensor = self.outfully, name = 'InputRNN' )

        # RNN Layers

        self.brain.addLayer( input        = 'InputRNN',
                             type         = tb.layers.rnn,
                             cell_type    = 'LSTM',
                             num_cells    = 2,
                             out_channels = 256,
                             name         = 'RNN')

        # Fully Connected Layers

        self.brain.setLayerDefaults( type          = tb.layers.fully,
                                     weight_stddev = 0.01 ,
                                     bias_stddev   = 0.01 )

        self.brain.addLayer( out_channels = self.num_actions,
                             activation   = tb.activs.softmax,
                             name         = 'Output' )
