from players_imitation.player_GAIL_1 import *


##### PLAYER GAIL
class player_GAIL_1A( player_GAIL_1 ):

    A_LEARNING_RATE = 1e-6
    C_LEARNING_RATE = 1e-3
    D_LEARNING_RATE = 1e-4

    NUM_FRAMES  = 1
    UPDATE_SIZE = 5
    BATCH_SIZE  = 128

    EPSILON = 0.10
    GAMMA   = 0.995
    LAM     = 0.97

    B_CLONING        = True # True to start with good policy
    BC_LEARNING_RATE = 1e-4
    INIT_NO_EPOCHS   = 50
    BC_BATCH_SIZE    = 50

    DS_SIZE        = 100000
    DATASET        = 'cartpole'

    ### __INIT__
    def __init__( self ):

        player_GAIL_1.__init__( self )

    # PROCESS OBSERVATION
    def process(self, obsv):

        return np.stack( tuple( self.obsv_list[i] for i in range( self.NUM_FRAMES ) ), axis = 1 )

    ### PREPARE NETWORK
    def network( self ):

        # Critic

        Critic = self.brain.addBlock( 'Critic' )

        Critic.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES ], name='Observation' )

        Critic.setLayerDefaults( type       = tb.layers.hlfully,
                                 activation = tb.activs.tanh )

        Critic.addLayer( out_channels = 256, input = 'Observation' )
        #Critic.addLayer( out_channels = 200 )
        Critic.addLayer( out_channels = 1, name = 'Value', activation = None )

        # Actor

        Actor = self.brain.addBlock( 'Actor' )

        Actor.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES ], name = 'Observation' )

        Actor.setLayerDefaults( type       = tb.layers.hlfully,
                                activation = tb.activs.tanh )

        Actor.addLayer( out_channels = 256 , input = 'Observation',  name = 'Hidden' )
        #Actor.addLayer( out_channels = 200,  name = 'Hidden' )

        Actor.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = None, name = 'Mu')
        Actor.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softplus, name = 'Sigma', activation_pars = 0.001 )
        Actor.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softmax,  name = 'Discrete' )

        mu     = Actor.tensor( 'Mu' )
        sigma  = Actor.tensor( 'Sigma' )
        dist   = tb.extras.dist_normal( mu, sigma )
        action = tb.aux.tf_squeeze( dist.sample( 1 ), 0 )
        Actor.addInput( tensor = action, name = 'Output')

        # OldActor

        Old = self.brain.addBlock( 'Old' )

        Old.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES ], name = 'Observation' )

        Old.setLayerDefaults( type       = tb.layers.hlfully,
                              activation = tb.activs.tanh )

        Old.addLayer( out_channels = 256 , input = 'Observation',  name = 'Hidden' )
        #Old.addLayer( out_channels = 200, name = 'Hidden' )

        Old.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = None, name = 'Mu')
        Old.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softplus, name = 'Sigma', activation_pars = 0.001 )
        Old.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softmax,  name = 'Discrete' )

        # Discriminator

        Disc = self.brain.addBlock( 'Disc' )

        Disc.addInput( shape = [ None, self.obsv_shape[0], self.NUM_FRAMES  ], name = 'Observation' )
        Disc.addInput( shape = [ None, self.num_actions   ], name = 'Action' )

        Disc.addLayer( input = 'Observation', type = tb.layers.flatten, name = 'ObservationFlat' )

        traj = tb.aux.tf_concat( Disc.tensor( 'ObservationFlat' ), Disc.tensor( 'Action' ), 1 )
        Disc.addInput( tensor = traj, name = 'Trajectory')

        Disc.setLayerDefaults( type = tb.layers.hlfully,
                               activation = tb.activs.tanh,
                               weight_stddev = 0.01,
                               bias_stddev   = 0.01 )

        Disc.addLayer( out_channels = 256, input = 'Trajectory',  name = 'Hidden' )
        #Disc.addLayer( out_channels = 200, name = 'Hidden' )
        Disc.addLayer( out_channels = 1,  name = 'Output', activation =  None)

        logits = Disc.tensor( 'Output' )
        reward = tb.extras.log_sig( logits )
        Disc.addInput( tensor = reward, name = 'DiscRew')
