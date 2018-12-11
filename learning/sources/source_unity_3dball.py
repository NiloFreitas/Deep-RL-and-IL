from sources.source_unity import source_unity
import numpy as np


##### SOURCE UNITY 3D BALL
class source_unity_3dball( source_unity ):

    ### __INIT__
    def __init__( self ):

        source_unity.__init__( self , "3dball" )
        self.range_act = 2

    ### INFORMATION
    def range_actions( self ): return self.range_act

    ### MAP KEYS
    def map_keys( self , actn ):

        return np.clip( actn, -self.range_act, self.range_act)

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        return obsv
