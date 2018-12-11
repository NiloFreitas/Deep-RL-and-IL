from players_reinforcement.player_dql_bayesian_2 import *


# PLAYER DQL BAYESIAN
class player_dql_bayesian_2A( player_dql_bayesian_2 ):

    NUM_FRAMES = 2
    BATCH_SIZE = 512

    LEARNING_RATE   = 3e-4
    REWARD_DISCOUNT = 0.99

    START_RANDOM_PROB        = 1.00
    FINAL_RANDOM_PROB        = 0.05
    NUM_EXPLORATION_EPISODES = 200

    EXPERIENCES_LEN    = 100000
    STEPS_BEFORE_TRAIN = 1000

    ### __INIT__
    def __init__( self ):

        player_dql_bayesian_2.__init__( self )

    # PROCESS OBSERVATION
    def process(self, obsv):

        return np.stack( tuple( self.obsv_list[i] for i in range(self.NUM_FRAMES) ), axis=2 )

    # PREPARE NETWORK
    def network(self):

        # Input Placeholder

        self.brain.addInput(shape=[None, self.obsv_shape[0], self.obsv_shape[1], self.NUM_FRAMES],
                            name='Observation')

        # Convolutional Layers

        self.brain.setLayerDefaults( type          = tb.layers.conv2d,
                                     activation    = tb.activs.relu,
                                     pooling       = 2,
                                     weight_stddev = 0.01,
                                     bias_stddev   = 0.01)

        self.brain.addLayer( out_channels=32, ksize=8, strides=4, input='Observation' )
        self.brain.addLayer( out_channels=64, ksize=4, strides=2 )
        self.brain.addLayer( out_channels=64, ksize=3, strides=1 )

        # Fully Connected Layers

        self.brain.setLayerDefaults( type       = tb.layers.fully,
                                     activation = tb.activs.relu )

        self.brain.addLayer( out_channels = 256,
                             dropout      = True,
                             dropout_name = 'Drop')

        self.brain.addLayer( out_channels = self.num_actions,
                             activation   = None,
                             name         = 'Output')
