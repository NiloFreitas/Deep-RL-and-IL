from sources.source_gym import source_gym
import numpy as np


##### SOURCE GYM HOPPER
class source_gym_hopper( source_gym ):

    ### __INIT__
    def __init__( self ):

        source_gym.__init__( self , 'Hopper-v2' )

    ### INFORMATION
    def num_actions( self ): return self.env.action_space.shape[0]
    def range_actions( self ): return abs(self.env.action_space.high[0])

    ### MAP KEYS
    def map_keys( self , actn ):

        actn = np.clip( actn, self.env.action_space.low[0], self.env.action_space.high[0])
        return np.expand_dims(actn,0)

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        return obsv
