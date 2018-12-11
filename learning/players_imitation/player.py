import random
from collections import deque

import sys
sys.path.append( '..' )
sys.path.append( '../..' )
import tensorblock as tb
import numpy as np

##### PLAYER
class player:

    ### __INIT__
    def __init__( self ):

        self.num_stored_obsv = 0

    ### DUMMY FUNCTIONS
    def prepare( self ): return None
    def network( self ): return None
    def operations( self ): return None
    def train( self , prev_state , curr_obsv , actn ,rewd , done ): return None
    def process( self , obsv ): return obsv
    def act( self , state ): return None
    def info( self ): return None
    def on_start( self ): return None

    ### START
    def start( self , source , obsv ):

        self.obsv_list = deque()
        self.obsv_shape = obsv.shape
        self.num_actions = source.num_actions()
        self.range_actions = source.range_actions()

        self.continuous = False
        if source.range_actions() != -1:
            self.continuous = True

        self.initialize()
        return self.restart( obsv )

    ### RESTART
    def restart( self , obsv ):

        self.obsv_list.clear()
        self.store_obsv( obsv )
        return self.process( obsv )

    ### PARSE ARGUMENTS
    def parse_args( self , args ):

        self.arg_save = args.save[0]
        self.arg_save_step = int( args.save[1] ) if len( args.save ) > 1 else 10
        self.arg_load = args.load
        self.arg_run = args.run

        if self.arg_save == 'same' : self.arg_save = self.arg_load
        if self.arg_load == 'same' : self.arg_load = self.arg_save

        if self.arg_save is not None : self.arg_save = '../trained_models/' + self.arg_save
        if self.arg_load is not None : self.arg_load = '../trained_models/' + self.arg_load

    ### CREATE ACTION
    def create_action( self , idx ):

        if self.continuous: return idx

        action = np.zeros( self.num_actions )
        action[ idx ] = 1

        return action

    ### CREATE RANDOM ACTION
    def create_random_action( self ):

        return self.create_action( random.randint( 0 , self.num_actions - 1 ) )
        #return self.env.action_space.sample()

    ### STORE OBSERVATIONS
    def store_obsv( self , obsv ):

        if self.num_stored_obsv > 0:

            while len( self.obsv_list ) < self.num_stored_obsv + 1:
                self.obsv_list.append( obsv )
            self.obsv_list.popleft()

    ### INITIALIZE NETWORK
    def initialize( self ):

        self.brain = tb.recipe()

        self.network()
        self.operations()

        if self.arg_save is not None:
            self.brain.addSaver( name = 'Save' , dir = self.arg_save )
        if self.arg_load is not None:
            self.brain.addSaver( name = 'Load' , dir = self.arg_load )

        self.brain.initialize()
        if self.arg_load is not None:
            print( '*** RESTORING' , self.arg_load )
            self.brain.restore( name = 'Load' )

        if not self.arg_run:
            self.on_start()

    ### LEARN FROM CURRENT OBSERVATION
    def learn( self , prev_state , curr_obsv , actn , rewd , done, episode ):

        self.store_obsv( curr_obsv )
        curr_state = self.process( curr_obsv )
        if not self.arg_run:
            self.train( prev_state, curr_state , actn , rewd , done, episode )

        return curr_state

    ### TIME TO SAVE
    def time_to_save( self , episode ):
        return not self.arg_run and self.arg_save is not None and \
               ( episode + 1 ) % self.arg_save_step == 0

    ### VERBOSE OUTPUT
    def verbose( self , episode , rewd , done ):

        if done:

            self.info()

            if self.time_to_save( episode ):
                print( ' SAVING |' , end = '' )
                self.brain.save( name = 'Save' )
            print()
