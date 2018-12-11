from players_reinforcement.player_dql_bayesian_1 import *


# PLAYER DQL BAYESIAN
class player_dql_bayesian_1A( player_dql_bayesian_1 ):

    NUM_FRAMES = 3
    BATCH_SIZE = 50

    LEARNING_RATE   = 1e-4
    REWARD_DISCOUNT = 0.99

    START_RANDOM_PROB        = 1.0
    FINAL_RANDOM_PROB        = 0.05
    NUM_EXPLORATION_EPISODES = 200

    EXPERIENCES_LEN    = 100000
    STEPS_BEFORE_TRAIN = 150

    ### __INIT__
    def __init__( self ):

        player_dql_bayesian_1.__init__( self )

    # PROCESS OBSERVATION
    def process(self, obsv):

        return np.stack( tuple( self.obsv_list[i] for i in range(self.NUM_FRAMES) ), axis=1 )

    # PREPARE NETWORK
    def network(self):

        # Input Placeholder

        self.brain.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES ],
                             name  = 'Observation' )

        # Fully Connected Layers

        self.brain.setLayerDefaults( type       = tb.layers.fully,
                                     activation = tb.activs.relu )

        self.brain.addLayer( out_channels = 512,
                             dropout      = True,
                             dropout_name = 'Drop',
                             name = 'Hidden' )

        self.brain.addLayer( input = 'Hidden',
                             out_channels = self.num_actions,
                             activation   = None,
                             name         = 'Output')
