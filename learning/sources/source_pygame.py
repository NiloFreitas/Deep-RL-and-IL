import signal
import sys
import importlib
from sources.source import source


##### SOURCE PYGAME
class source_pygame( source ):

    ### __INIT__
    def __init__( self , game ):

        source.__init__( self )
        module = importlib.import_module( 'sources.pygames.' + game )
        self.env = getattr( module , game )()

        def signal_handler(signal, frame):
            print('\nProgram closed!')
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    ### START SIMULATION
    def start( self ):

        obsv = self.env.reset()
        return self.process( obsv )

    ### MOVE ONE STEP
    def move( self , actn ):

        obsv , rewd , done = self.env.step( self.map_keys( actn ) )
        return self.process( obsv ) , rewd , done
