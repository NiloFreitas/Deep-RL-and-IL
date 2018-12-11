from players_imitation.player_DAgger_1 import *


# PLAYER DAgger
class player_DAgger_1A( player_DAgger_1 ):

    LEARNING_RATE  = 1e-4
    BETA           = 0.5

    NUM_FRAMES     = 1
    BATCH_SIZE     = 64
    EPOCHS         = 10
    TIME_TO_UPDATE = 5000

    DS_SIZE        = 100000
    DATASET        = 'cartpole'

    ### __INIT__
    def __init__( self ):

        player_DAgger_1.__init__( self )

    # PROCESS OBSERVATION
    def process(self, obsv):

        return np.stack( tuple( self.obsv_list[i] for i in range( self.NUM_FRAMES ) ), axis = 1 )

    ### PREPARE NETWORK
    def network( self ):

        # Expert Policy

        Expert = self.brain.addBlock( 'Expert' )

        Expert.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES ], name = 'Observation' )

        Expert.setLayerDefaults( type       = tb.layers.fully,
                                 activation = tb.activs.relu )

        Expert.addLayer( out_channels = 256, input = 'Observation', name = 'Hidden'  )
        #Expert.addLayer( out_channels = 200, name = 'Hidden' )
        Expert.addLayer( out_channels = self.num_actions, input = 'Hidden', name = 'Output',   activation = None )
        Expert.addLayer( out_channels = self.num_actions, input = 'Hidden', name = 'Discrete', activation = tb.activs.softmax )

        # Actor Policy

        Actor = self.brain.addBlock( 'Actor' )

        Actor.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES ], name = 'Observation' )

        Actor.setLayerDefaults( type       = tb.layers.fully,
                                activation = tb.activs.relu )

        Actor.addLayer( out_channels = 256, input = 'Observation', name = 'Hidden' )
        #Actor.addLayer( out_channels = 200, name = 'Hidden' )
        Actor.addLayer( out_channels = self.num_actions, input = 'Hidden', name = 'Output',   activation = None )
        Actor.addLayer( out_channels = self.num_actions, input = 'Hidden', name = 'Discrete', activation = tb.activs.softmax )
