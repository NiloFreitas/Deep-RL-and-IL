from players_reinforcement.player import player
from auxiliar.aux_plot import *
import tensorflow as tf

import random
from collections import deque

import sys
sys.path.append('..')
import tensorblock as tb
import numpy as np


# PLAYER A2C
class player_A2C_1(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.experiences     = deque()
        self.num_stored_obsv = self.NUM_FRAMES

    # CHOOSE NEXT ACTION
    def act(self, state):

        return self.calculate( state )

    # CALCULATE NETWORK
    def calculate(self, state):

        output = np.squeeze( self.brain.run( 'Output', [ [ 'Observation', [state] ] ] ) )
        action = np.random.choice( np.arange(len(output)), p=output )
        return self.create_action( action )

    # PREPARE NETWORK

    def operations(self):

        # Action Placeholders

        self.brain.addInput( shape = [ None, self.num_actions ], name = 'Actions' )
        self.brain.addInput( shape = [ None, 1                ], name = 'Advantage' )

        # Operations

            # Critic

        self.brain.addOperation( function = tb.ops.mean_squared_error,
                                 input    = ['Value','Advantage'],
                                 name     = 'CriticCost')

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'CriticCost',
                                 learning_rate = self.C_LEARNING_RATE,
                                 name          = 'CriticOptimizer' )

            # Actor

        self.brain.addOperation( function = tb.ops.pgcost,
                                 input    = [ 'Output', 'Actions', 'Advantage' ],
                                 name     = 'ActorCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'ActorCost',
                                 learning_rate = self.A_LEARNING_RATE,
                                 summary       = 'Summary',
                                 writer        = 'Writer',
                                 name          = 'ActorOptimizer' )

        # TensorBoard

        self.brain.addSummaryScalar( input = 'ActorCost' )
        self.brain.addWriter( name = 'Writer' , dir = '../' )
        self.brain.addSummary( name = 'Summary' )
        self.brain.initialize()

    # TRAIN NETWORK
    def train(self, prev_state, curr_state, actn, rewd, done, episode):

        # Store New Experience

        self.experiences.append( ( prev_state , curr_state , actn , rewd , done ) )
        if len( self.experiences ) > self.EXPERIENCES_LEN: self.experiences.popleft()

        # Check for Train

        if len( self.experiences ) > self.STEPS_BEFORE_TRAIN and self.BATCH_SIZE > 0:

            # Select Random Batch

            batch = random.sample( self.experiences , self.BATCH_SIZE )

            # Separate Batch Data

            prev_states = [d[0] for d in batch]
            curr_states = [d[1] for d in batch]
            actions     = [d[2] for d in batch]
            rewards     = [d[3] for d in batch]
            dones       = [d[4] for d in batch]

            # States Value

            prev_values = np.squeeze(self.brain.run( 'Value' , [ [ 'Observation' , prev_states ] ] ) )
            next_values = np.squeeze(self.brain.run( 'Value' , [ [ 'Observation' , curr_states ] ] ) )

            # Calculate TD Targets and TD Errors

            td_targets = []
            td_errors  = []
            for i in range( len(rewards) ):
                if dones[i]:
                    td_targets.append ( rewards[i] )
                    td_errors.append  ( td_targets[i] - prev_values[i] )
                else:
                    td_targets.append ( rewards[i] + self.REWARD_DISCOUNT * next_values[i] )
                    td_errors.append  ( td_targets[i] - prev_values[i] )

            td_targets = np.expand_dims( td_targets, 1 )
            td_errors  = np.expand_dims( td_errors,  1 )

            # Optimize Neural Network

            _, = self.brain.run( ['CriticOptimizer'], [ [ 'Observation', prev_states ],
                                                        [ 'Advantage',   td_targets  ] ] )

            _, c, summary = self.brain.run( [ 'ActorOptimizer','ActorCost','Summary' ], [ [ 'Observation', prev_states ],
                                                                                          [ 'Actions',     actions     ],
                                                                                          [ 'Advantage',   td_errors   ] ] )

            # TensorBoard

            self.brain.write( summary = summary, iter = episode )
