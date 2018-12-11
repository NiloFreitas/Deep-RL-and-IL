from players_reinforcement.player import player
from auxiliar.aux_plot import *

import sys
sys.path.append('..')

import tensorblock as tb
import numpy as np


# PLAYER PPO
class player_PPO_1(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.experiences = []

    # CHOOSE NEXT ACTION
    def act(self, state):

        return self.calculate(state)

    # CALCULATE NETWORK
    def calculate(self, state):

        if self.continuous:
            action = self.brain.run( 'Actor/Output', [ [ 'Actor/Observation', [state] ] ] )
            action = np.reshape( action, self.num_actions )

        if not self.continuous:
            output = np.squeeze( self.brain.run( 'Actor/Discrete', [ [ 'Actor/Observation', [state] ] ] ) )
            action = np.random.choice( np.arange( len( output ) ), p = output )

        return self.create_action( action )

    # PREPARE NETWORK
    def operations(self):

        # Placeholders

        self.brain.addInput( shape = [ None, self.num_actions ], name = 'Actions'  )
        self.brain.addInput( shape = [ None, self.num_actions ], name = 'O_Mu'  )
        self.brain.addInput( shape = [ None, self.num_actions ], name = 'O_Sigma'  )
        self.brain.addInput( shape = [ None, self.num_actions ], name = 'O_Discrete'  )
        self.brain.addInput( shape = [ None, 1 ] ,               name = 'Advantage')

        # Operations

            # Critic

        self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                 input    = [ 'Critic/Value','Advantage' ],
                                 name     = 'CriticCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'CriticCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'CriticOptimizer' )
            # Actor

        if self.continuous:
            self.brain.addOperation( function = tb.ops.ppocost_distrib,
                                     input    = [ 'Actor/Mu',
                                                  'Actor/Sigma',
                                                  'O_Mu',
                                                  'O_Sigma',
                                                  'Actions',
                                                  'Advantage',
                                                  self.EPSILON ],
                                     name     = 'ActorCost' )

        if not self.continuous:
            self.brain.addOperation( function = tb.ops.ppocost,
                                     input    = [ 'Actor/Discrete',
                                                  'O_Discrete',
                                                  'Actions',
                                                  'Advantage',
                                                  self.EPSILON ],
                                     name     = 'ActorCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'ActorCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'ActorOptimizer' )

            # Assign Old Actor

        self.brain.addOperation( function = tb.ops.assign,
                                 input = ['Old', 'Actor'],
                                 name = 'Assign' )

    # TRAIN NETWORK
    def train( self, prev_state, curr_state, actn, rewd, done, episode ):

        # Store New Experience Until Train

        self.experiences.append( (prev_state, curr_state, actn, rewd, done) )

        # Check for Train

        if ( len(self.experiences) >= self.BATCH_SIZE ):

            batch = self.experiences

            # Separate Batch Data

            prev_states = [d[0] for d in batch]
            curr_states = [d[1] for d in batch]
            actions     = [d[2] for d in batch]
            rewards     = [d[3] for d in batch]
            dones       = [d[4] for d in batch]

            # States Values

            prev_values = np.squeeze( self.brain.run( 'Critic/Value' , [ [ 'Critic/Observation', prev_states  ] ] ) )
            curr_values = np.squeeze( self.brain.run( 'Critic/Value' , [ [ 'Critic/Observation', curr_states  ] ] ) )

            # Calculate Generalized Advantage Estimation

            running_add_y = 0
            running_add_a = 0
            y = np.zeros_like(rewards)
            advantage  = rewards + (self.GAMMA * curr_values) - prev_values
            for t in reversed ( range( 0, len( advantage ) ) ):
                if dones[t]:
                    curr_values[t] = 0
                    running_add_a  = 0
                running_add_y  = curr_values[t] * self.GAMMA            + rewards   [t]
                running_add_a  = running_add_a  * self.GAMMA * self.LAM + advantage [t]
                y [t] = running_add_y
                advantage [t] = running_add_a
            y = np.expand_dims( y, 1 )
            advantage = np.expand_dims( advantage, 1 )

            # Assign Old Pi

            self.brain.run( ['Assign'], [] )

            # Get Old Probabilities

            if self.continuous:
                o_Mu, o_Sigma = self.brain.run( [ 'Old/Mu', 'Old/Sigma' ], [ [ 'Old/Observation', prev_states ] ] )

            if not self.continuous:
                o_Discrete =  self.brain.run(  'Old/Discrete' , [ [ 'Old/Observation', prev_states ] ] )

            # Optimize

            for _ in range (self.UPDATE_SIZE):

                if self.continuous:
                    self.brain.run( [ 'ActorOptimizer' ], [ [ 'Actor/Observation',  prev_states ],
                                                            [ 'O_Mu',               o_Mu        ],
                                                            [ 'O_Sigma',            o_Sigma     ],
                                                            [ 'Actions',            actions     ],
                                                            [ 'Advantage',          advantage   ] ] )
                if not self.continuous:
                    self.brain.run( [ 'ActorOptimizer' ], [ [ 'Actor/Observation',  prev_states ],
                                                            [ 'O_Discrete',         o_Discrete  ],
                                                            [ 'Actions',            actions     ],
                                                            [ 'Advantage',          advantage   ] ] )

                self.brain.run( [ 'CriticOptimizer' ], [ [ 'Critic/Observation', prev_states ],
                                                         [ 'Advantage',          y           ] ] )

            # Reset

            self.experiences = []
