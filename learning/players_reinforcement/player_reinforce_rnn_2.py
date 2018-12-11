from players_reinforcement.player import player
from auxiliar.aux_plot import *

import random
from collections import deque

import sys
sys.path.append('..')
import tensorblock as tb
import numpy as np

import tensorflow as tf


# PLAYER REINFORCE RNN
class player_reinforce_rnn_2(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)
        self.experiences = deque()

    # CHOOSE NEXT ACTION
    def act(self, state):

        return self.calculate(state)

    # CALCULATE NETWORK
    def calculate(self, state):

        size = len( self.experiences )

        if size < self.NUM_FRAMES:
            return self.create_random_action()

        states = np.zeros( (self.NUM_FRAMES , self.obsv_shape[0], self.obsv_shape[1] ) )

        for i , j in enumerate( range( size - self.NUM_FRAMES , size  ) ):
            states[i] = self.experiences[j][1]

        states = np.expand_dims( states, 0 )
        output = np.squeeze( self.brain.run('Output', [['Observation', states]]) )
        action = np.random.choice( np.arange(len(output)), p=output )

        return self.create_action(action)

    # PREPARE NETWORK
    def operations(self):

        # Action Placeholders

        self.brain.addInput( shape = [ None , self.num_actions ] , name = 'Actions' )
        self.brain.addInput( shape = [ None                    ] , name = 'Target'  )


        # Operations

        self.brain.addOperation( function = tb.ops.pgcost,
                                 input    = [ 'Output', 'Actions', 'Target' ],
                                 name     = 'Cost' )

        # Optimizer

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'Cost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'Optimizer' )

        # TensorBoard

        self.brain.addSummaryScalar( input = 'Cost' )
        self.brain.addSummaryHistogram( input = 'Target' )
        self.brain.addWriter( name = 'Writer' , dir = '../' )
        self.brain.addSummary( name = 'Summary' )
        self.brain.initialize()

    # TRAIN NETWORK
    def train(self, prev_state, curr_state, actn, rewd, done, episode):

        # Store New Experience Until Done

        self.experiences.append((prev_state, curr_state, actn, rewd, done))

        batchsize = len( self.experiences ) - self.NUM_FRAMES + 1

        # Check for Train

        if done:

            # Select Batch

            batch = self.experiences

            # Separate Batch Data

            prev_states = np.zeros( ( batchsize , self.NUM_FRAMES , self.obsv_shape[0], self.obsv_shape[1] ) )
            curr_states = np.zeros( ( batchsize , self.NUM_FRAMES , self.obsv_shape[0], self.obsv_shape[1] ) )
            actions     = np.zeros( ( batchsize , self.num_actions ) )
            rewards     = np.zeros( ( batchsize ) )
            dones       = np.zeros( ( batchsize ) )

            # Select Batches

            for i in range( 0 , batchsize ):

                for j in range( 0 , self.NUM_FRAMES ):

                    prev_states[i,j,:,:] = self.experiences[ i + j ][0]
                    curr_states[i,j,:,:] = self.experiences[ i + j ][1]

                actions[i] = self.experiences[ i + self.NUM_FRAMES  - 1][2]
                rewards[i] = self.experiences[ i + self.NUM_FRAMES  - 1][3]
                dones[i]   = self.experiences[ i + self.NUM_FRAMES  - 1][4]

            # Calculate Discounted Reward

            running_add = 0
            discounted_r = np.zeros_like(rewards)
            for t in reversed(range(0, len(rewards))):
                if rewards[t] != 0:  # pygame_catch specific
                    running_add = 0
                running_add = running_add * self.REWARD_DISCOUNT + rewards[t]
                discounted_r[t] = running_add

            # Optimize Neural Network

            _, summary = self.brain.run( ['Optimizer','Summary'], [ ['Observation', prev_states ],
                                                                    ['Actions',  actions        ],
                                                                    ['Target', discounted_r     ] ] )

            # TensorBoard

            self.brain.write( summary = summary, iter = episode )

            # Reset Batch

            self.experiences = deque()
