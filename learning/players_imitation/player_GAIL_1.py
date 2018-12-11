from players_imitation.player import player

import sys
sys.path.append('..')

import tensorblock as tb
import numpy as np
import random

# PLAYER GAIL
class player_GAIL_1(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.num_stored_obsv = self.NUM_FRAMES
        self.experiences = []
        self.s_dataset = []
        self.a_dataset = []

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

        # Action Placeholders

        self.brain.addInput( shape = [ None, self.num_actions ], name = 'Actions'    )
        self.brain.addInput( shape = [ None, self.num_actions ], name = 'O_Mu'       )
        self.brain.addInput( shape = [ None, self.num_actions ], name = 'O_Sigma'    )
        self.brain.addInput( shape = [ None, self.num_actions ], name = 'O_Discrete' )
        self.brain.addInput( shape = [ None, 1 ],                name = 'Advantage'  )
        self.brain.addInput( shape = [ None, 1 ],                name = 'Exp_Logits' )

        # Operations

            # Critic

        self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                 input    = [ 'Critic/Value','Advantage' ],
                                 name     = 'CriticCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'CriticCost',
                                 learning_rate = self.A_LEARNING_RATE,
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
                                 learning_rate = self.A_LEARNING_RATE,
                                 name          = 'ActorOptimizer' )

            # Discriminator

        self.brain.addOperation( function = tb.ops.disccost,
                                 input    = [ 'Disc/Output', 'Exp_Logits' ],
                                 name     = 'DiscCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'DiscCost',
                                 learning_rate = self.D_LEARNING_RATE,
                                 name          = 'DiscOptimizer' )

            # Assign

        self.brain.addOperation( function = tb.ops.assign,
                                 input    = [ 'Old', 'Actor' ],
                                 name     = 'Assign' )

            # Behaviour Cloning

        if self.continuous:
            self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                     input    = [ 'Actor/Output', 'Actions' ],
                                        name     = 'BcCost' )
        if not self.continuous:
            self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                     input    = [ 'Actor/Discrete', 'Actions' ],
                                     name     = 'BcCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'BcCost',
                                 learning_rate = self.BC_LEARNING_RATE,
                                 name          = 'BcOptimizer' )

    # RUN ONCE ON START
    def on_start( self ):

        self.s_dataset = np.load( '../datasets/' + self.DATASET + '_states.npy'  )[ 0 : self.DS_SIZE ]
        self.a_dataset = np.load( '../datasets/' + self.DATASET + '_actions.npy' )[ 0 : self.DS_SIZE ]

        self.s_dataset = np.expand_dims(self.s_dataset,2)

        # Start with Behaviour Cloning

        if self.B_CLONING:

            for _ in range ( self.INIT_NO_EPOCHS * len(self.s_dataset) // self.BC_BATCH_SIZE ):

                idx = np.random.randint(len(self.s_dataset), size = self.BC_BATCH_SIZE)

                states  = self.s_dataset[idx,:]
                actions = self.a_dataset[idx,:]

                self.brain.run( [ 'BcOptimizer' ], [ [ 'Actor/Observation', states  ],
                                                     [ 'Actions',           actions ] ] )

            self.brain.run( ['Assign'], [] )

    # TRAIN NETWORK
    def train( self, prev_state, curr_state, actn, rewd, done, episode ):

        # Store New Experience Until Done

        self.experiences.append( (prev_state, curr_state, actn, rewd, done) )

        # Check for Train

        if (len(self.experiences)) >= self.BATCH_SIZE:

            # Select Batch

            batch = self.experiences

            # Separate Batch Data

            prev_states = [d[0] for d in batch]
            curr_states = [d[1] for d in batch]
            actions     = [d[2] for d in batch]
            #rewards    = [d[3] for d in batch]
            dones       = [d[4] for d in batch]

            # Trajectories Evaluation

            idx = np.random.randint( self.DS_SIZE - 2 * self.BATCH_SIZE, size = 1 )
            #idxs = np.random.choice(self.DS_SIZE, size=self.BATCH_SIZE)
            idxs = np.arange( idx, idx + self.BATCH_SIZE )
            #idxs2 = np.random.choice(200, size=self.BATCH_SIZE)

            #prev_states = np.array(prev_states)[idxs2,:,:]
            #curr_states = np.array(curr_states)[idxs2,:,:]
            #actions     = np.array(actions)[idxs2,:]
            #dones       = np.array(dones)[idxs2]

            s_dataset = self.s_dataset[idxs,:]
            a_dataset = self.a_dataset[idxs,:]
            #s_dataset = np.expand_dims(s_dataset,2)
            #a_dataset = np.expand_dims(a_dataset,2)
            exp_logits = self.brain.run( 'Disc/Output' , [ [ 'Disc/Observation', s_dataset ],
                                                           [ 'Disc/Action',      a_dataset ] ] )

            # Update Discriminator

            self.brain.run( [ 'DiscOptimizer' ], [ [ 'Disc/Observation', prev_states ],
                                                   [ 'Disc/Action',      actions     ],
                                                   [ 'Exp_Logits',       exp_logits  ] ] )
            # States Values

            prev_values = np.squeeze( self.brain.run( 'Critic/Value' , [ [ 'Critic/Observation', prev_states  ] ] ) )
            curr_values = np.squeeze( self.brain.run( 'Critic/Value' , [ [ 'Critic/Observation', curr_states  ] ] ) )

            # Discriminator Rewards

            rewards = np.squeeze( self.brain.run( 'Disc/DiscRew' , [ [ 'Disc/Observation', prev_states ],
                                                                     [ 'Disc/Action',      actions     ] ] ) )

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

            # Update Old Pi

            self.brain.run( ['Assign'], [] )

            # Get Old Probabilities

            if self.continuous:
                o_Mu, o_Sigma = self.brain.run( [ 'Old/Mu', 'Old/Sigma' ], [ [ 'Old/Observation', prev_states ] ] )

            if not self.continuous:
                o_Discrete =  self.brain.run(  'Old/Discrete' , [ [ 'Old/Observation', prev_states ] ] )

            # Update Actor and Critic

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
