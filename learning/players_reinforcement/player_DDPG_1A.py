from players_reinforcement.player_DDPG_1 import *
import tensorflow as tf
import numpy as np


# PLAYER DDPG
class player_DDPG_1A( player_DDPG_1 ):

    EXPERIENCES_LEN    = 1e6
    BATCH_SIZE         = 128
    STEPS_BEFORE_TRAIN = 5000
    NUM_FRAMES         = 1
    C_LEARNING_RATE    = 1e-3
    A_LEARNING_RATE    = 1e-4
    TAU                = 0.01
    REWARD_DISCOUNT    = 0.99

    ### __INIT__
    def __init__( self ):

        player_DDPG_1.__init__( self )

    # PROCESS OBSERVATION
    def process(self, obsv):

        return np.stack( tuple(self.obsv_list[i] for i in range(self.NUM_FRAMES)), axis = 1 )

    # PREPARE NETWORK
    def network( self ):

        # Critic

        NormalCritic = self.brain.addBlock( 'NormalCritic' )

        NormalCritic.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES ], name  = 'Observation' )
        NormalCritic.addInput( shape = [ None, self.num_actions ], name  = 'Actions' )

        NormalCritic.setLayerDefaults( type       = tb.layers.hlfully,
                                       activation = tb.activs.relu,
                                       bias       = False)

        NormalCritic.addLayer( input = 'Observation',out_channels = 128, name = 'Hidden1' )

        H1 = NormalCritic.tensor( 'Hidden1' )
        H2 = NormalCritic.tensor( 'Actions' )
        H3 = tf.concat( [H1,H2], 1 )
        NormalCritic.addInput( tensor = H3, name = 'Hidden3' )

        NormalCritic.addLayer( input = 'Hidden3', out_channels = 200, name = 'Hidden4' )
        NormalCritic.addLayer( input = 'Hidden4', out_channels = 1, activation = None , name = 'Value' )

        # Target Critic

        TargetCritic = self.brain.addBlock( 'TargetCritic' )

        TargetCritic.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES ], name  = 'Observation')
        TargetCritic.addInput( shape = [ None, self.num_actions ], name  = 'Actions' )

        TargetCritic.setLayerDefaults( type       = tb.layers.hlfully,
                                       activation = tb.activs.relu,
                                       bias       = False )

        TargetCritic.addLayer( input = 'Observation', out_channels = 128, name = 'Hidden1', copy_weights = '../NormalCritic/W_Hidden1' )

        H1 = TargetCritic.tensor ('Hidden1' )
        H2 = TargetCritic.tensor( 'Actions' )
        H3 = tf.concat( [H1,H2],1 )
        TargetCritic.addInput( tensor = H3, name = 'Hidden3')

        TargetCritic.addLayer( input = 'Hidden3', out_channels = 200, name = 'Hidden4', copy_weights = '../NormalCritic/W_Hidden4' )
        TargetCritic.addLayer( input = 'Hidden4', out_channels = 1, activation = None , name = 'Value', copy_weights = '../NormalCritic/W_Value' )

        # Actor

        NormalActor = self.brain.addBlock( 'NormalActor' )

        NormalActor.addInput( shape=[ None, self.obsv_shape[0], self.NUM_FRAMES ], name='Observation' )

        NormalActor.setLayerDefaults( type       = tb.layers.hlfully,
                                      activation = tb.activs.relu,
                                      bias       = False )

        NormalActor.addLayer( input = 'Observation', out_channels = 128, name = 'Hidden1' )
        NormalActor.addLayer( input = 'Hidden1',     out_channels = 200, name = 'Hidden2' )
        NormalActor.addLayer( input = 'Hidden2',     out_channels = self.num_actions,
                              activation = tb.activs.tanh, name = 'Out', min = -0.003, max = 0.003)

        out = NormalActor.tensor('Out')
        out = tf.multiply( out, self.range_actions )
        NormalActor.addInput( tensor = out, name = 'Output')

        # Target Actor

        TargetActor = self.brain.addBlock( 'TargetActor' )

        TargetActor.addInput( shape=[ None, self.obsv_shape[0], self.NUM_FRAMES ], name='Observation' )

        TargetActor.setLayerDefaults( type       = tb.layers.hlfully,
                                      activation = tb.activs.relu,
                                      bias       = False )

        TargetActor.addLayer( input = 'Observation', out_channels = 128, name = 'Hidden1', copy_weights = '../NormalActor/W_Hidden1')
        TargetActor.addLayer( input = 'Hidden1',     out_channels = 200, name = 'Hidden2', copy_weights = '../NormalActor/W_Hidden2')
        TargetActor.addLayer( input = 'Hidden2',     out_channels = self.num_actions, name = 'Out',
                              activation = tb.activs.tanh, min = -0.003, max = 0.003, copy_weights = '../NormalActor/W_Out')

        out = TargetActor.tensor('Out')
        out = tf.multiply( out, self.range_actions )
        TargetActor.addInput( tensor = out, name = 'Output')
