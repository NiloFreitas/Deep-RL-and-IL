from players_reinforcement.player import player
from auxiliar.aux_plot import *
import tensorflow as tf

import random
from collections import deque

import sys
sys.path.append('..')
import tensorblock as tb
import numpy as np


# PLAYER DDPG
class player_DDPG_1(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.experiences     = deque()
        self.num_stored_obsv = self.NUM_FRAMES
        self.noise_state     = 0
        self.dt              = 0.01

    ## ORNSTEIN-UHLENBECK PROCESS
    def OU( self, mu, theta, sigma):

        x = self.noise_state
        dx =  self.dt * theta * (mu - self.noise_state) + sigma * np.random.randn(self.num_actions) *  np.sqrt(self.dt)
        self.noise_state = x + dx

        return self.noise_state

    ### CHOOSE NEXT ACTION
    def act( self , state):

        return self.calculate( state )

    # CALCULATE NETWORK
    def calculate(self, state):

        action = self.brain.run( 'NormalActor/Output', [ [ 'NormalActor/Observation', [state] ] ] )
        noise  = self.OU( mu = 0, theta = 0.15 , sigma = 0.2 )
        action = action[0] + noise

        return self.create_action( np.reshape( action, [self.num_actions] ) )

    # PREPARE NETWORK
    def operations(self):

        # Placeholders

        self.brain.addInput( shape = [ None, 1                ], name = 'TDTarget',    dtype = tf.float32 )
        self.brain.addInput( shape = [ None, self.num_actions ], name = 'ActionGrads', dtype = tf.float32 )

        # Operations

            # Critic

        self.brain.addOperation( function = tb.ops.get_grads,
                                 input = [ 'NormalCritic/Value', 'NormalCritic/Actions' ],
                                 summary  = 'Summary',
                                 writer   = 'Writer',
                                 name     = 'GetGrads' )

        self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                 input    = [ 'NormalCritic/Value', 'TDTarget' ],
                                 name     = 'CriticCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'CriticCost',
                                 learning_rate = self.C_LEARNING_RATE,
                                 name          = 'CriticOptimizer' )
            # Actor

        self.brain.addOperation( function = tb.ops.combine_grads,
                                 input    = [ 'NormalActor/Output', 'ActionGrads' ],
                                 name     = 'CombineGrads' )

        self.brain.addOperation( function      = tb.optims.adam_apply,
                                 input         = [ 'CombineGrads' ],
                                 learning_rate = self.A_LEARNING_RATE,
                                 name          = 'ActorOptimizer' )

            # Assign Softly

        self.brain.addOperation( function = tb.ops.assign_soft,
                                 input = ['TargetCritic', 'NormalCritic', self.TAU],
                                 name = 'AssignCritic')

        self.brain.addOperation( function = tb.ops.assign_soft,
                                 input = ['TargetActor',   'NormalActor', self.TAU],
                                 name = 'AssignActor')

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

            # States Values

            target_actns = self.brain.run( 'TargetActor/Output', [ [ 'TargetActor/Observation',  curr_states  ] ] )
            next_values  = self.brain.run( 'TargetCritic/Value', [ [ 'TargetCritic/Observation', curr_states  ],
                                                                   [ 'TargetCritic/Actions',     target_actns ] ] )

            # Calculate Expected Reward

            expected_rewards = []
            for i in range( self.BATCH_SIZE ):
                if dones[i]:
                     expected_rewards.append( rewards[i] )
                else:
                    expected_rewards.append( rewards[i] + self.REWARD_DISCOUNT * next_values[i] )

            expected_rewards = np.reshape( expected_rewards, [ self.BATCH_SIZE, 1 ] )

            # Optimize Critic

            _ = self.brain.run( ['CriticOptimizer'], [ [ 'NormalCritic/Observation', prev_states      ],
                                                       [ 'NormalCritic/Actions',     actions          ],
                                                       [ 'TDTarget',                 expected_rewards ] ] )
            # Get New Actions

            new_a = self.brain.run( 'NormalActor/Output', [ ['NormalActor/Observation', prev_states ] ] )

            # Get Critic Grads wrt New Actions

            grads = self.brain.run( ['GetGrads'], [ [ 'NormalCritic/Observation', prev_states],
                                                    [ 'NormalCritic/Actions',     new_a      ] ] )

            # Optimize Actor

            _ = self.brain.run( ['ActorOptimizer'], [ [ 'NormalActor/Observation', prev_states ],
                                                      [ 'ActionGrads',             grads[0]    ] ] )

            # Copy weights to Target Networks

            self.brain.run( ['AssignActor'], [] )
            self.brain.run( ['AssignCritic'], [] )
