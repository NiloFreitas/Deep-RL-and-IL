import signal
import sys
import gym
from sources.source import source
from gym import wrappers

##### SOURCE GYM
class source_gym( source ):

    render = True 

    ### __INIT__
    def __init__( self , game ):

        source.__init__( self )
        self.env = gym.make( game )
        #self.env = wrappers.Monitor(self.env, ".") #record

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

        obsv , rewd , done, info = self.env.step( self.map_keys( actn ) )
        if self.render: self.env.render()
        return self.process( obsv ) , rewd , done
