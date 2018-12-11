from players_reinforcement.player import player
from auxiliar.aux_plot import *

import random
from collections import deque

import sys
sys.path.append('..')
import tensorblock as tb
import numpy as np


# PLAYER REINFORCE
class player_reinforce_2(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.num_stored_obsv = self.NUM_FRAMES
        self.experiences     = deque()

    # CHOOSE NEXT ACTION
    def act(self, state):

        return self.calculate(state)

    # CALCULATE NETWORK
    def calculate(self, state):

        output = np.squeeze( self.brain.run( 'Output', [ [ 'Observation', [state] ] ] ) )
        action = np.random.choice( np.arange(len(output)), p=output )
        return self.create_action( action )

    # PREPARE NETWORK
    def operations(self):

        # Action Placeholders

        self.brain.addInput( shape = [ None, self.num_actions ], name = 'Actions' )
        self.brain.addInput( shape = [ None                   ], name = 'Target' )

        # Operations

        self.brain.addOperation( function = tb.ops.pgcost,
                                 input    = [ 'Output', 'Actions', 'Target' ],
                                 name     = 'Cost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'Cost',
                                 learning_rate = self.LEARNING_RATE,
                                 summary       = 'Summary',
                                 writer        = 'Writer',
                                 name          = 'Optimizer' )

        # TensorBoard

        self.brain.addSummaryScalar( input = 'Cost' )
        self.brain.addSummaryHistogram( input = 'Target' )
        self.brain.addWriter( name = 'Writer' , dir = '../' )
        self.brain.addSummary( name = 'Summary' )
        self.brain.initialize()

    # TRAIN NETWORK
    def train( self, prev_state, curr_state, actn, rewd, done, episode ):

        # Store New Experience Until Done

        self.experiences.append( ( prev_state, curr_state, actn, rewd, done ) )

        # Check for Train

        if done:

            # Select Batch

            batch = self.experiences

            # Separate Batch Data

            prev_states = [d[0] for d in batch]
            curr_states = [d[1] for d in batch]
            actions     = [d[2] for d in batch]
            rewards     = [d[3] for d in batch]
            dones       = [d[4] for d in batch]

            # Calculate Discounted Reward

            running_add = 0
            discounted_r = np.zeros_like(rewards)
            for t in reversed(range(0, len(rewards))):
                if rewards[t] !=0:  # pygame_catch specific
                    running_add = 0
                running_add = running_add * self.REWARD_DISCOUNT + rewards[t]
                discounted_r[t] = running_add

            # Optimize Neural Network

            _, summary = self.brain.run( ['Optimizer','Summary'], [ [ 'Observation', prev_states  ],
                                                                    [ 'Actions',     actions      ],
                                                                    [ 'Target',      discounted_r ] ] )

            # TensorBoard

            self.brain.write( summary = summary, iter = episode )

            # Reset

            self.experiences = deque()
