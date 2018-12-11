import argparse
import importlib
import numpy as np

# Parse args
parser = argparse.ArgumentParser( description = 'Input Arguments' )
parser.add_argument( 'type' ,   nargs = 1 ) # reinforcement or imitation
parser.add_argument( 'inputs' , nargs = 2 ) # player and source
parser.add_argument( '--load' ,  dest = 'load' ,  default = None )
parser.add_argument( '--save' ,  dest = 'save' ,  default = [ None ] , nargs = '*' )
parser.add_argument( '--epis' ,  dest = 'epis' ,  default = 1e9 )
parser.add_argument( '--run'  ,  dest = 'run'  ,  action  = 'store_true' )
parser.add_argument( '--record', dest = 'record', action  = 'store_true'  )
args = parser.parse_args()

# Import modules
type_string = args.type[0]
source_string , player_string = args.inputs
source_module = importlib.import_module( 'sources.' + source_string )
player_module = importlib.import_module( 'players_' + type_string + '.' + player_string )

# Get instances
source = getattr( source_module , source_string )()
player = getattr( player_module , player_string )()
player.parse_args( args )

# Start source and player
obsv = source.start()
state = player.start( source , obsv )

# Define some variables
states_buffer, actions_buffer = [],[]
done = False
episode, step, n_episodes = 0, 0, int( args.epis )

# Learning loop
while episode < n_episodes:

    # Run a step and get info
    actn = player.act( state )                     # Choose Next Action
    obsv , rewd , env_done = source.move( actn )   # Run Next Action On Source
    step +=1

    # For metric/plotting only, episode defined as 1000 steps
    if step % 1000 == 0: done = True

    # Save trajectories if recording
    if args.record:
        states_buffer.append( state )
        actions_buffer.append( actn )
        if done:
            np.save("datasets/states",  np.array(states_buffer),  allow_pickle=True, fix_imports=True)
            np.save("datasets/actions", np.array(actions_buffer), allow_pickle=True, fix_imports=True)

    # Train the algorithm
    state = player.learn( state , obsv , actn , rewd , env_done, episode )  # Learn From This Action

    # Verbose
    source.verbose( episode , rewd , done ) # Source Text Output
    player.verbose( episode , rewd , done ) # Player Text Output

    # If environment episode is over
    if env_done:
        obsv = source.start()
        state = player.restart( obsv )

    # If metric episode is over
    if done:
        episode += 1
        done = False
