# Credits to the code in 'sources/carla/Environment': https://github.com/GokulNC/Setting-Up-CARLA-RL
from sources.carla.Environment.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv
from sources.source import source
import signal
import sys
import cv2


##### SOURCE CARLA
class source_carla( source ):

    ### __INIT__
    def __init__( self ):

        source.__init__( self )

        self.continuous = False

        # Discrete actions (9):
        #   int: 0:NONE, 1:TURN_LEFT, 2:TURN_RIGHT, 3:GAS, 4:BRAKE, 5:GAS_AND_TURN_LEFT,
        #        6:GAS_AND_TURN_RIGHT, 7:BRAKE_AND_TURN_LEFT, 8:BRAKE_AND_TURN_RIGHT

        # Continuous actions (3):
        #   list: [throttle_value, steering_value, brake_value]

        self.env = CarlaEnv( is_render_enabled = False,
                             num_speedup_steps = 10,
                             run_offscreen = False,
                             cameras = ['SceneFinal', 'Depth', 'SemanticSegmentation'],
                             save_screens = False,
                             continuous = self.continuous )

        def signal_handler(signal, frame):
            print('\nProgram closed!')
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

    ### INFORMATION
    def num_actions( self ):
        if self.continuous: return 3
        else: return 9

    def range_actions( self ):
        if self.continuous: return 1
        else: return -1

    ### START SIMULATION
    def start( self ):

        self.action = [1,0,0,0,0,0,0,0,0]
        if self.continuous: self.action = [0,0,0]

        self.env.reset()

        observation, reward, done, _ = self.env.step( self.map_keys(self.action) )

        car_speed        = observation['forward_speed']
        car_acceleration = observation['acceleration']
        rgb_image        = observation['rgb_image']

        return self.process( rgb_image )

    ### MOVE ONE STEP
    def move( self , actn ):

        observation, reward, done, _ = self.env.step( self.map_keys(actn) )

        car_speed        = observation['forward_speed']
        car_acceleration = observation['acceleration']
        rgb_image        = observation['rgb_image']

        return self.process( rgb_image ), reward, done

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        obsv = cv2.resize( obsv , ( 84 , 84 ) )

        return obsv

    ### MAP KEYS
    def map_keys( self , actn ):

        if self.continuous:
            return actn

        else:
            if actn[0] : return 0
            if actn[1] : return 1
            if actn[2] : return 2
            if actn[3] : return 3
            if actn[4] : return 4
            if actn[5] : return 5
            if actn[6] : return 6
            if actn[7] : return 7
            if actn[8] : return 8
