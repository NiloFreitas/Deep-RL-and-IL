from players_imitation.player import player

import sys
sys.path.append('..')

import tensorblock as tb
import numpy as np
import random

# PLAYER DAgger
class player_DAgger_1(player):

    # __INIT__
    def __init__(self):

        player.__init__(self)

        self.num_stored_obsv = self.NUM_FRAMES
        self.experiences = []
        self.s_dataset = []
        self.a_dataset = []

        self.iteration   = 0
        self.first_run   = True

    # CHOOSE NEXT ACTION
    def act(self, state):

        return self.calculate(state)

    # CALCULATE NETWORK
    def calculate(self, state):

        if self.first_run:
            BETA = 1
        else:
            BETA = self.BETA

        # Actor Actions

        if random.random() > BETA:

            if self.continuous:
                action =  self.brain.run( 'Actor/Output', [ [ 'Actor/Observation', [state] ] ] )
                action = np.reshape( action, self.num_actions )

            if not self.continuous:
                action = np.squeeze( self.brain.run( 'Actor/Discrete', [ [ 'Actor/Observation', [state] ] ] ) )
                action = np.random.choice( np.arange( len( action ) ), p = action )

    # Expert Actions

        else:

            if self.continuous:
                action = self.brain.run( 'Expert/Output', [ [ 'Expert/Observation', [state] ] ] )
                action = np.reshape( action, self.num_actions )

            if not self.continuous:
                action = np.squeeze( self.brain.run( 'Expert/Discrete', [ [ 'Expert/Observation', [state] ] ] ) )
                action = np.random.choice( np.arange( len( action ) ), p = action )

        return self.create_action( action )

    # PREPARE NETWORK
    def operations(self):

        # Placeholders

        self.brain.addInput( shape = [ None, self.num_actions ], name = 'Labels' )

        # Operations

            # Actor

        if self.continuous:
            self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                     input    = [ 'Actor/Output', 'Labels' ],
                                     name     = 'ActorCost' )

        if not self.continuous:
            self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                     input    = [ 'Actor/Discrete', 'Labels' ],
                                     name     = 'ActorCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'ActorCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'ActorOptimizer' )

            # Expert

        if self.continuous:
            self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                     input    = [ 'Expert/Output', 'Labels' ],
                                     name     = 'ExpertCost' )

        if not self.continuous:
            self.brain.addOperation( function = tb.ops.hlmean_squared_error,
                                     input    = [ 'Expert/Discrete', 'Labels' ],
                                     name     = 'ExpertCost' )

        self.brain.addOperation( function      = tb.optims.adam,
                                 input         = 'ExpertCost',
                                 learning_rate = self.LEARNING_RATE,
                                 name          = 'ExpertOptimizer' )

    # RUN ONCE ON START
    def on_start( self ):

        self.s_dataset = np.load('../datasets/' + self.DATASET + '_states.npy'  )[ 0 : self.DS_SIZE ]
        self.a_dataset = np.load('../datasets/' + self.DATASET + '_actions.npy' )[ 0 : self.DS_SIZE ]

        #self.s_dataset = np.expand_dims(self.s_dataset,2)

        # Train Expert Once

        for _ in range ( self.EPOCHS * len(self.s_dataset) // self.BATCH_SIZE ):

            idx = np.random.randint(len(self.s_dataset), size = self.BATCH_SIZE)

            states  = self.s_dataset[idx,:]
            actions = self.a_dataset[idx,:]

            self.brain.run( [ 'ExpertOptimizer' ], [ [ 'Expert/Observation',  states  ],
                                                     [ 'Labels',              actions ] ] )

    # TRAIN NETWORK
    def train( self, prev_state, curr_state, actn, rewd, done, episode ):

        # Store New Experience Until Done

        self.experiences.append( (prev_state, curr_state, actn, rewd, done) )

        # Check for Aggregate and Train

        if ( len(self.experiences) >= self.TIME_TO_UPDATE ):

            # Change Beta

            self.iteration += 1

            self.BETA **= self.iteration

            # Select Batch

            batch = self.experiences

            # Separate Batch Data

            prev_states = [d[0] for d in batch]
            curr_states = [d[1] for d in batch]
            #actions    = [d[2] for d in batch]
            rewards     = [d[3] for d in batch]
            dones       = [d[4] for d in batch]

            # Calculate Expert Actions

            exp_actions = self.brain.run( 'Expert/Output', [ [ 'Expert/Observation', prev_states ] ] )

            # Aggregate Datasets

            self.s_dataset = np.concatenate ( (self.s_dataset, prev_states ), axis = 0 )
            self.a_dataset = np.concatenate ( (self.a_dataset, exp_actions ), axis = 0 )

            # Train Actor

            for _ in range ( self.EPOCHS * len(self.s_dataset) // self.BATCH_SIZE ):

                idx = np.random.randint(len(self.s_dataset), size = self.BATCH_SIZE)

                states  = self.s_dataset[idx,:]
                actions = self.a_dataset[idx,:]

                self.brain.run( [ 'ActorOptimizer' ], [ [ 'Actor/Observation',  states      ],
                                                        [ 'Labels',             actions ] ] )

            # Reset

            self.experiences = []
            self.first_run = False
