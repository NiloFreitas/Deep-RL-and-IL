
from sources.source_gym import source_gym
import cv2
import numpy as np


##### SOURCE GYM CARTPOLE
class source_gym_cartpole( source_gym ):

    ### __INIT__
    def __init__( self ):

        source_gym.__init__( self , 'CartPole-v0' )

    ### INFORMATION
    def num_actions( self ): return self.env.action_space.n

    ### MAP KEYS
    def map_keys( self , actn ):

        if actn[0] : return 0
        if actn[1] : return 1

    ### MOVE ONE STEP
    def move( self , actn ):

        obsv , rewd , done, info = self.env.step( self.map_keys( actn ) )
        if done: rewd -= 50

        if self.render: self.env.render()
        return self.process( obsv ) , rewd , done

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        return obsv
