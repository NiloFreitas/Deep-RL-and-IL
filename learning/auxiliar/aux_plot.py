import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

### PLOT STATE
def plot_state( state ):

    n = state.shape[2]

    fig = plt.figure()
    for i in range( n ):
        plt.subplot( 2 , n , i + 1 )
        plt.imshow( state[:,:,i] , cmap = 'gray' )
    plt.show()

### PLOT STATES
def plot_states( prev_state , curr_state ):

    n = prev_state.shape[2]

    fig = plt.figure()
    for i in range( n ):
        plt.subplot( 2 , n , i + 1 )
        plt.imshow( curr_state[:,:,i] , cmap = 'gray' )
        plt.subplot( 2 , n , i + n + 1 )
        plt.imshow( prev_state[:,:,i] , cmap = 'gray' )
    plt.show()


### SAVE STATISTICS
def plot_episode_stats(episode_lengths, episode_rewards, accumulated_lenghts, time_rewards):

    np.savetxt('./auxiliar/EpisodeLengths.txt', episode_lengths,     fmt='%.5f', newline='\n')
    np.savetxt('./auxiliar/EpisodeRewards.txt', episode_rewards,     fmt='%.5f', newline='\n')
    np.savetxt('./auxiliar/AccLengths.txt',     accumulated_lenghts, fmt='%.5f', newline='\n')
    np.savetxt('./auxiliar/TimeRewards.txt',    time_rewards,        fmt='%.5f', newline='\n')

    # Plot the episode length over time
    fig1 = plt.figure(figsize=(20, 10))

    plt.subplot(221)
    plt.plot(episode_rewards, color = 'c')

    plt.xlabel("Steps (thousands)")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Steps")

    plt.subplot(222)
    plt.plot(accumulated_lenghts, episode_rewards, color = 'g')

    plt.xlabel("Time (seconds)")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Time")

    plt.subplot(223)
    plt.plot(episode_lengths, color = 'r')

    plt.xlabel("Steps (thousands)")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Steps")

    plt.subplot(224)
    plt.plot(accumulated_lenghts, color = 'm')

    plt.xlabel("Steps (thousands)")
    plt.ylabel("Accumulated Length")
    plt.title("Accumulated Length over Steps")

    plt.savefig('./auxiliar/Plot.png')

    plt.close()
