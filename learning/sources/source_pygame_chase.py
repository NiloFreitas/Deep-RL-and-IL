from sources.source_pygame import source_pygame


##### SOURCE PYGAME CHASE
class source_pygame_chase( source_pygame ):

    ### __INIT__
    def __init__( self ):

        source_pygame.__init__( self , 'pygame_chase' )

    ### INFORMATION
    def num_actions( self ): return 5

    ### MAP KEYS
    def map_keys( self , actn ):

        if actn[0] : return 0
        if actn[1] : return 1
        if actn[2] : return 2
        if actn[3] : return 3
        if actn[4] : return 4

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        return obsv
