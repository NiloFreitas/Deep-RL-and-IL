from sources.vrep.vrepper.core import vrepper
from sources.source import source

import sys, os, time, signal
import numpy as np

##### SOURCE VREP
class source_vrep( source ):

    ### __INIT__
    def __init__( self, scene ):

        source.__init__( self )

        self.env = vrepper( headless = not self.RENDER )
        self.env.start()
        self.env.load_scene(os.path.dirname(os.path.realpath(__file__)) + '/vrep/scenes/' + scene + '.ttt')

        def signal_handler(signal, frame):
            print('\nProgram closed!')
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

    ### START SIMULATION
    def start( self ):

        self.INITIAL_POSITION = np.array([0,0,0,0,0,0]) #np.random.uniform(-90, 90, 6)

        self.env.start_simulation(is_sync=False)
        time.sleep(.2)

        for i, m in enumerate(self.objects):
            if i == 0:
                self.INITIAL_POSITION[i] *= -1
            m.force_position(self.INITIAL_POSITION[i])
        time.sleep(.2)

        self.env.make_simulation_synchronous(True)

        # Get first observation
        obsv, rewd, done = self._get_obsv(self.DESIRED_POSITION)

        return obsv

    ### MOVE ONE STEP
    def move( self , actn ):

        # Act
        self.step( self.map_keys(actn) )
        self.env.step_blocking_simulation()

        # Get observation
        obsv, rewd, done = self._get_obsv(self.DESIRED_POSITION)

        return obsv, rewd, done

    ### CHILD METHODS
    def get_obsv(self, desired_position):
        raise NotImplementedError

    def step(self, positions, speeds=None):
        raise NotImplementedError
