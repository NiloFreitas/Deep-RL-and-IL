from sources.source import source
from sources.carla.control import *
import signal
import sys
import cv2
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


##### SOURCE CARLA
class source_carla( source ):

    # Discrete actions:
        # 0 - Throttle
        # 1 - Throttle and right steer
        # 2 - Throttle and left steer
        # 3 - Brake

    ### __INIT__
    def __init__( self ):

        source.__init__( self )

        class Args:

            debug = True
            host = '127.0.0.1'
            port = 2000
            autopilot = False
            res = '600x400'
            width, height = [int(x) for x in res.split('x')]

        self.args = Args()
        self.env = CarlaEnv()

        # Open Server
        self.env.open_server(self.args)
        # Open Client
        self.env.init(self.args)

        def signal_handler(signal, frame):
            print('\nProgram closed!')
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

    ### INFORMATION
    def num_actions( self ):
        return 4

    def range_actions( self ):
        return -1

    ### START SIMULATION
    def start( self ):

        obsv, rewd, done = self.env.step([0,0,0,0])

        return self.process(obsv)

    ### MOVE ONE STEP
    def move( self , actn ):

        obsv, rewd, done = self.env.step(actn)

        return self.process(obsv), rewd, done

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        # Convert image to gray
        obsv = np.uint8(obsv)
        obsv = cv2.resize( obsv , ( 84 , 84 ) )
        obsv = cv2.cvtColor( obsv , cv2.COLOR_BGR2GRAY )

        # Plot the ANN image input:
        #fig = plt.figure()
        #for i in range( 1 ):
        #    plt.subplot( 2 , 1 , i + 1 )
        #    plt.imshow( obsv[:,:] , cmap = 'gray' )
        #plt.savefig('./auxiliar/rgb.png')
        #plt.close()

        return obsv
