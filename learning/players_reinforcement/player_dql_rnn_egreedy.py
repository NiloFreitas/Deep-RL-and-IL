import random
from collections import deque
from players_reinforcement.player import player

import sys
sys.path.append( '..' )
import tensorblock as tb
import numpy as np

import time

##### PLAYER DQL RNN EGREEDY
class player_dql_rnn_egreedy( player ):

    ### __INIT__
    def __init__( self ):

        player.__init__( self )

        self.random_prob = self.START_RANDOM_PROB

        self.experiences = deque()
        self.sequence = []

    ### INFO ON SCREEN
    def info( self ):

        if self.random_prob > 0.0 and not self.arg_run:
            print( ' Random : %5.5f |' % ( self.random_prob ) , end = '' )

        if self.random_prob > self.FINAL_RANDOM_PROB:
            self.random_prob -= ( self.START_RANDOM_PROB - self.FINAL_RANDOM_PROB ) / self.NUM_EXPLORATION_EPISODES

    ### CHOOSE NEXT ACTION
    def act( self , state ):

        if self.random_prob == 0.0 or self.arg_run:
            return self.calculate( state )

        if random.random() > self.random_prob:
            return self.calculate( state )
        else:
            return self.create_random_action()

    ### CALCULATE NETWORK
    def calculate( self , state ):

        size = len( self.sequence )

        if size < self.NUM_FRAMES:
            return self.create_random_action()

        states = np.zeros( ( self.NUM_FRAMES , self.obsv_shape[0] ) )
        for i , j in enumerate( range( size - self.NUM_FRAMES , size ) ):
            states[i] = self.sequence[j][1]

        output = self.brain.run( 'Output' , [ [ 'Observation' , [ states ] ] ] )
        return self.create_action( np.argmax( output ) )

    ### PREPARE OPERATIONS
    def operations( self ):

        # Action Placeholders

        self.brain.addInput( shape = [ None , self.num_actions ] , name = 'Actions' )
        self.brain.addInput( shape = [ None                    ] , name = 'Target'  )

        # Operations

        self.brain.addOperation( function = tb.ops.sum_mul ,
                                 input = [ 'Output' , 'Actions' ] , name = 'Readout' )
        self.brain.addOperation( function = tb.ops.mean_squared_error ,
                                 input = [ 'Target' , 'Readout' ] , name = 'Cost' )

        # Optimizer

        self.brain.addOperation( function = tb.optims.adam , input = 'Cost' ,
                                 learning_rate = self.LEARNING_RATE , name  = 'Optimizer' )

    ### TRAIN NETWORK
    def train( self, prev_state, curr_state, actn, rewd, done, episode ):

        # Store New Experience

        self.sequence.append( ( prev_state , curr_state , actn , rewd , done ) )
        if done: self.experiences.append( self.sequence ) ; self.sequence = []

        # Check for Train
        if len( self.experiences ) >= self.STEPS_BEFORE_TRAIN and self.BATCH_SIZE > 0:

            # Creat Batch Structures

            prev_states = np.zeros( ( self.BATCH_SIZE , self.NUM_FRAMES , self.obsv_shape[0] ) )
            curr_states = np.zeros( ( self.BATCH_SIZE , self.NUM_FRAMES , self.obsv_shape[0] ) )
            actions     = np.zeros( ( self.BATCH_SIZE , self.num_actions ) )
            rewards     = np.zeros( ( self.BATCH_SIZE ) )
            dones       = np.zeros( ( self.BATCH_SIZE ) )

            # Select Random Batches

            for i in range( 0 , self.BATCH_SIZE ):

                rnd_i = random.randint( 0 , len( self.experiences ) - 1 )
                rnd_j = random.randint( 0 , len( self.experiences[rnd_i] ) - self.NUM_FRAMES )

                for j in range( 0 , self.NUM_FRAMES ):

                    prev_states[i,j,:] = self.experiences[ rnd_i ][ rnd_j + j ][0]
                    curr_states[i,j,:] = self.experiences[ rnd_i ][ rnd_j + j ][1]

                actions[i] = self.experiences[ rnd_i ][ rnd_j + self.NUM_FRAMES - 1 ][2]
                rewards[i] = self.experiences[ rnd_i ][ rnd_j + self.NUM_FRAMES - 1 ][3]
                dones[i]   = self.experiences[ rnd_i ][ rnd_j + self.NUM_FRAMES - 1 ][4]

            # Calculate Rewards for each Action

            rewards_per_action = self.brain.run( 'Output' , [ [ 'Observation' , curr_states ] ] )

            # Calculate Expected Reward

            expected_reward = []
            for i in range( self.BATCH_SIZE ):
                if dones[i]: expected_reward.append( rewards[i] )
                else:        expected_reward.append( rewards[i] +
                                self.REWARD_DISCOUNT * np.max( rewards_per_action[i] ) )

            # Optimize Neural Network

            self.brain.run( 'Optimizer' , [ [ 'Observation' , prev_states     ] ,
                                            [ 'Actions'     , actions         ] ,
                                            [ 'Target'      , expected_reward ] ] )
