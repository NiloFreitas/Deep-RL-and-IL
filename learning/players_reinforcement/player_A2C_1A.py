from players_reinforcement.player_A2C_1 import *
import tensorflow as tf


# PLAYER A2C
class player_A2C_1A( player_A2C_1 ):

    BATCH_SIZE         = 50
    NUM_FRAMES         = 3
    C_LEARNING_RATE    = 1e-4
    A_LEARNING_RATE    = 1e-5
    REWARD_DISCOUNT    = 0.99
    EXPERIENCES_LEN    = 1e6
    STEPS_BEFORE_TRAIN = 500

    ### __INIT__
    def __init__( self ):

        player_A2C_1.__init__( self )

    # PROCESS OBSERVATION
    def process(self, obsv):

        return np.stack( tuple( self.obsv_list[i] for i in range( self.NUM_FRAMES ) ), axis = 1 )

    # PREPARE NETWORK
    def network( self ):

        # Input Placeholder

        self.brain.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES ],
                             name  = 'Observation')

        # Critic -------------------------------------------------------------

        # Fully Connected Layers

        self.brain.setLayerDefaults( type       = tb.layers.fully,
                                     activation = tb.activs.elu,
                                     weight_stddev = 0.01,
                                     bias_stddev   = 0.01 )

        self.brain.addLayer( input = 'Observation', out_channels = 512 )
        self.brain.addLayer( out_channels = 512 )
        self.brain.addLayer( out_channels = 1, activation = None ,name = 'Value' )

        # Actor --------------------------------------------------------------

        # Fully Connected Layers

        self.brain.setLayerDefaults( type       = tb.layers.fully,
                                     activation = tb.activs.elu,
                                     weight_stddev = 0.01,
                                     bias_stddev   = 0.01 )

        self.brain.addLayer( input = 'Observation', out_channels = 512 )
        self.brain.addLayer( out_channels = 512 )
        self.brain.addLayer( out_channels = self.num_actions, activation = tb.activs.softmax, name = 'Output' )
