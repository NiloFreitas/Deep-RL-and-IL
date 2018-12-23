from players_reinforcement.player_PPO_2 import *


# PLAYER PPO
class player_PPO_2A( player_PPO_2 ):

    NUM_FRAMES    = 3
    LEARNING_RATE = 3e-4
    UPDATE_SIZE   = 5
    BATCH_SIZE    = 256
    EPSILON       = 0.2
    GAMMA         = 0.99
    LAM           = 0.95
    rgb           = 1 # 1 if black and white

    ### __INIT__
    def __init__( self ):

        player_PPO_2.__init__( self )

    # PROCESS OBSERVATION
    def process(self, obsv):

        obsv = np.stack( tuple( self.obsv_list[i] for i in range( self.NUM_FRAMES ) ), axis = -1 )

        if self.rgb > 1: obsv = obsv.reshape(-1,self.obsv_shape[0],self.obsv_shape[1],self.NUM_FRAMES * self.rgb)[0]

        return obsv

    ### PREPARE NETWORK
    def network( self ):

        # Critic

        Critic = self.brain.addBlock( 'Critic' )

        Critic.addInput( shape = [ None, self.obsv_shape[0], self.obsv_shape[1], self.NUM_FRAMES * self.rgb ],
                         name  = 'Observation' )

        Critic.setLayerDefaults( type          = tb.layers.conv2d,
                                 activation    = tb.activs.relu,
                                 pooling       = 2,
                                 weight_stddev = 0.01,
                                 bias_stddev   = 0.01 )

        Critic.addLayer( out_channels = 32, ksize = 8, strides = 4, input = 'Observation' )
        Critic.addLayer( out_channels = 64, ksize = 4, strides = 2 )
        Critic.addLayer( out_channels = 64, ksize = 3, strides = 1 )

        Critic.setLayerDefaults( type       = tb.layers.fully,
                                 activation = tb.activs.relu )

        Critic.addLayer( out_channels = 512 )
        Critic.addLayer( out_channels = 1, name = 'Value', activation = None )

        # Actor

        Actor = self.brain.addBlock( 'Actor' )

        Actor.addInput( shape = [ None, self.obsv_shape[0], self.obsv_shape[1], self.NUM_FRAMES * self.rgb ],
                         name  = 'Observation' )

        Actor.setLayerDefaults( type          = tb.layers.conv2d,
                                 activation    = tb.activs.relu,
                                 pooling       = 2,
                                 weight_stddev = 0.01,
                                 bias_stddev   = 0.01 )

        Actor.addLayer( out_channels = 32, ksize = 8, strides = 4, input = 'Observation' )
        Actor.addLayer( out_channels = 64, ksize = 4, strides = 2 )
        Actor.addLayer( out_channels = 64, ksize = 3, strides = 1 )

        Actor.setLayerDefaults( type       = tb.layers.fully,
                                activation = tb.activs.relu )

        Actor.addLayer( out_channels = 512, name = 'Hidden' )

        Actor.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = None, name = 'Mu')
        Actor.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softplus, name = 'Sigma', activation_pars = 0.5 )
        Actor.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softmax,  name = 'Discrete' )

        mu     = Actor.tensor( 'Mu' )
        sigma  = Actor.tensor( 'Sigma' )
        dist   = tb.extras.dist_normal( mu, sigma )
        action = dist.sample( 1 )
        Actor.addInput( tensor = action, name = 'Output')

        # OldActor

        Old = self.brain.addBlock( 'Old' )

        Old.addInput( shape = [ None, self.obsv_shape[0], self.obsv_shape[1], self.NUM_FRAMES * self.rgb ],
                         name  = 'Observation' )

        Old.setLayerDefaults( type          = tb.layers.conv2d,
                                 activation    = tb.activs.relu,
                                 pooling       = 2,
                                 weight_stddev = 0.01,
                                 bias_stddev   = 0.01 )

        Old.addLayer( out_channels = 32, ksize = 8, strides = 4, input = 'Observation' )
        Old.addLayer( out_channels = 64, ksize = 4, strides = 2 )
        Old.addLayer( out_channels = 64, ksize = 3, strides = 1 )

        Old.setLayerDefaults( type       = tb.layers.fully,
                              activation = tb.activs.relu )

        Old.addLayer( out_channels = 512,  name = 'Hidden' )

        Old.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = None, name = 'Mu')
        Old.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softplus, name = 'Sigma', activation_pars = 0.5 )
        Old.addLayer( out_channels = self.num_actions , input = 'Hidden', activation = tb.activs.softmax,  name = 'Discrete' )
