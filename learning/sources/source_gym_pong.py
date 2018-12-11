from sources.source_gym import source_gym
import cv2
import numpy as np


##### SOURCE GYM PONG
class source_gym_pong( source_gym ):

    ### __INIT__
    def __init__( self ):

        source_gym.__init__( self , 'Pong-v4' )

    ### INFORMATION
    def num_actions( self ): return 3

    ### MAP KEYS
    def map_keys( self , actn ):

        if actn[0] : return 1
        if actn[1] : return 2
        if actn[2] : return 3

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        obsv = cv2.resize( obsv , ( 80 , 80 ) )
        obsv = cv2.cvtColor( obsv , cv2.COLOR_BGR2GRAY )
        _ , obsv = cv2.threshold( obsv , 97 , 255 , cv2.THRESH_BINARY )

        return obsv
