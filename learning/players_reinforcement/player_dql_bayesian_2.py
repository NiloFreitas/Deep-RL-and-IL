from players_reinforcement.player import player
from auxiliar.aux_plot import *

import random
from collections import deque

import sys
sys.path.append('..')
import tensorblock as tb
import numpy as np


# PLAYER DQL BAYESIAN
class player_dql_bayesian_2(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.num_stored_obsv = self.NUM_FRAMES
        self.random_prob     = self.START_RANDOM_PROB
        self.experiences     = deque()
        self.keep_prob       = self.FINAL_RANDOM_PROB

    # CHOOSE NEXT ACTION
    def act(self, state):

        return self.calculate(state)

    # CALCULATE NETWORK
    def calculate(self, state):

        output = self.brain.run( 'Output', [ [ 'Observation', [state] ] ],
                                 use_dropout=True )
        return self.create_action ( np.argmax(output) )

    # PREPARE OPERATIONS
    def operations ( self ):

        # Action Placeholders

        self.brain.addInput( shape = [ None, self.num_actions ], name = 'Actions' )
        self.brain.addInput( shape = [ None                   ], name = 'Target'  )

        # Operations

        self.brain.addOperation( function = tb.ops.sum_mul,
                                 input    = ['Output', 'Actions'],
                                 name     = 'Readout' )

        self.brain.addOperation( function = tb.ops.mean_squared_error,
                                 input    = ['Target', 'Readout'],
                                 name     = 'Cost' )

        # Optimizer

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'Cost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'Optimizer' )

        # TensorBoard

        self.brain.addSummaryScalar( input = 'Cost' )
        self.brain.addSummaryHistogram( input = 'Target' )
        self.brain.addWriter(name = 'Writer' , dir = '../' )
        self.brain.addSummary( name = 'Summary' )
        self.brain.initialize()

    # TRAIN NETWORK
    def train( self, prev_state, curr_state, actn, rewd, done, episode ):

        # Store New Experience

        self.experiences.append( ( prev_state, curr_state, actn, rewd, done ) )

        if len(self.experiences) > self.EXPERIENCES_LEN:
            self.experiences.popleft()

        # Check for Train

        if len(self.experiences) > self.STEPS_BEFORE_TRAIN and self.BATCH_SIZE > 0:

            # Select Random Batch

            batch = random.sample(self.experiences, self.BATCH_SIZE)

            # Separate Batch Data

            prev_states = [d[0] for d in batch]
            curr_states = [d[1] for d in batch]
            actions     = [d[2] for d in batch]
            rewards     = [d[3] for d in batch]
            dones       = [d[4] for d in batch]

            # Calculate Rewards

            rewards_per_action = self.brain.run( 'Output', [['Observation', curr_states]],
                                                 use_dropout = True )

            # Calculate Expected Reward

            expected_reward = []
            for i in range(self.BATCH_SIZE):
                if dones[i]:
                    expected_reward.append( rewards[i] )
                else:
                    expected_reward.append( rewards[i] + \
                                            self.REWARD_DISCOUNT * np.max( rewards_per_action[i] ) )

            # Optimize Neural Network

            _, summary = self.brain.run( ['Optimizer','Summary'], [ [ 'Observation', prev_states ],
                                                                    [ 'Actions', actions         ],
                                                                    [ 'Target', expected_reward  ] ], use_dropout=True )

            # Update Random Probability

            if done and self.random_prob > self.FINAL_RANDOM_PROB:
                self.random_prob -= (self.START_RANDOM_PROB - self.FINAL_RANDOM_PROB) / \
                    self.NUM_EXPLORATION_EPISODES
                self.keep_prob += (self.START_RANDOM_PROB - self.FINAL_RANDOM_PROB) / \
                    self.NUM_EXPLORATION_EPISODES

            self.brain.setDropout( name = 'Drop', value = self.keep_prob )

            # TensorBoard

            self.brain.write( summary = summary, iter = episode )
