
from sources.source_gym import source_gym
import cv2
import numpy as np

##### SOURCE GYM REACHER
class source_gym_reacher( source_gym ):

    ### __INIT__
    def __init__( self ):

        source_gym.__init__( self , 'Reacher-v2' )

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
