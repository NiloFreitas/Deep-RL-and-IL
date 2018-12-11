from sources.source_pygame import source_pygame
import cv2


##### SOURCE PYGAME CATCH
class source_pygame_catch( source_pygame ):

    ### __INIT__
    def __init__( self ):

        source_pygame.__init__( self , 'pygame_catch' )

    ### INFORMATION
    def num_actions( self ): return 3

    ### MAP KEYS
    def map_keys( self , actn ):

            if actn[0] : return 0
            if actn[1] : return 1
            if actn[2] : return 2

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        obsv = cv2.resize( obsv , ( 80 , 80 ) )
        obsv = cv2.cvtColor( obsv , cv2.COLOR_BGR2GRAY )
        _ , obsv = cv2.threshold( obsv , 127 , 255 , cv2.THRESH_BINARY )

        return obsv
