from players_reinforcement.player_reinforce_1 import *


# PLAYER REINFORCE
class player_reinforce_1A( player_reinforce_1 ):

    NUM_FRAMES      = 1
    LEARNING_RATE   = 1e-4
    REWARD_DISCOUNT = 0.99

    ### __INIT__
    def __init__( self ):

        player_reinforce_1.__init__( self )

    # PROCESS OBSERVATION
    def process(self, obsv):

        return np.stack( tuple( self.obsv_list[i] for i in range( self.NUM_FRAMES ) ), axis = 1 )

    # PREPARE NETWORK
    def network( self ):

        # Input Placeholder

        self.brain.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES ],
                             name  = 'Observation' )

        # Fully Connected Layers

        self.brain.setLayerDefaults( type       = tb.layers.fully ,
                                     activation = tb.activs.relu )

        self.brain.addLayer( out_channels = 64 ,
                             input        = 'Observation' )

        self.brain.addLayer( out_channels = 64 )

        self.brain.addLayer( out_channels = self.num_actions,
                             activation   = tb.activs.softmax,
                             name         = 'Output' )
