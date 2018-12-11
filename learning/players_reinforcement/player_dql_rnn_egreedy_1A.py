from players_reinforcement.player_dql_rnn_egreedy import *

##### PLAYER DQL RNN EGREEDY 1A
class player_dql_rnn_egreedy_1A( player_dql_rnn_egreedy ):

    NUM_FRAMES = 3
    BATCH_SIZE = 50

    LEARNING_RATE = 1e-4
    REWARD_DISCOUNT = 0.99

    START_RANDOM_PROB = 1.00
    FINAL_RANDOM_PROB = 0.05
    NUM_EXPLORATION_EPISODES = 250

    EXPERIENCES_LEN = 100000
    STEPS_BEFORE_TRAIN = 150

    ### __INIT__
    def __init__( self ):

        player_dql_rnn_egreedy.__init__( self )

    ### PREPARE NETWORK
    def network( self ):

        # Input Placeholder

        self.brain.addInput( shape = [ None , self.NUM_FRAMES , self.obsv_shape[0] ] ,
                             name = 'Observation' )

        # Fully Connected Layers

        self.brain.addLayer( type = tb.layers.rnn , input = 'Observation' , name = 'RNN' ,
                             num_cells = 1 , out_channels = 64 ,
                             activation = tb.activs.relu )

        self.brain.setLayerDefaults( type = tb.layers.fully ,
                                     activation = tb.activs.relu ,
                                     weight_stddev = 0.01 , bias_stddev = 0.01 )

        self.brain.addLayer( out_channels = 64 )
        self.brain.addLayer( out_channels = self.num_actions  ,
                             activation = None , name = 'Output' )
