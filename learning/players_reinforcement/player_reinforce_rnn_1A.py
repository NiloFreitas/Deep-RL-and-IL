from players_reinforcement.player_reinforce_rnn_1 import *
import tensorflow as tf


# PLAYER REINFORCE RNN
class player_reinforce_rnn_1A( player_reinforce_rnn_1 ):

    NUM_FRAMES      = 3
    LEARNING_RATE   = 1e-4
    REWARD_DISCOUNT = 0.99

    ### __INIT__
    def __init__( self ):

        player_reinforce_rnn_1.__init__( self )

    # PREPARE NETWORKx
    def network( self ):

        # Input Placeholder

        self.brain.addInput( shape = [ None, self.NUM_FRAMES, self.obsv_shape[0] ],
                             name  = 'Observation' )

        # RNN Layers

        self.brain.addLayer( type         = tb.layers.rnn,
                             cell_type    = 'LSTM',
                             num_cells    = 1,
                             out_channels = 64,
                             name         = 'RNN')

        # Fully Connected Layers

        self.brain.setLayerDefaults( type          = tb.layers.fully,
                                     weight_stddev = 0.01,
                                     bias_stddev   = 0.01 )

        self.brain.addLayer( type         = tb.layers.fully,
                             out_channels = 32 ,
                             activation   = tb.activs.relu )

        self.brain.addLayer( out_channels = self.num_actions,
                             activation   = tb.activs.softmax,
                             name         = 'Output' )
