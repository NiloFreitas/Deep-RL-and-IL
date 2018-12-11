import time
from collections import deque
from auxiliar.aux_plot import *


##### SOURCE
class source:

    ### __INIT__
    def __init__( self ):

        self.elapsed_time = 0
        self.avg_rewd = deque()
        self.sum_rewd = 0
        self.acc_rewd = 0
        self.acc_time = 0

        self.episode_rewards = deque()
        self.episode_lengths = deque()
        self.time_rewards = deque()
        self.accumulated_lenghts = deque()

        self.timer = time.time()

        return None

    ### DUMMY FUNCTIONS
    def num_actions( self ): return 0
    def range_actions( self ): return -1
    def map_keys( self , action ): return 0
    def process( self , obsv ): return obsv

    ### VERBOSE OUTPUT
    def verbose( self , episode , rewd , done , avg_length = 10 ):

        self.sum_rewd += rewd

        if done:

            self.acc_rewd += self.sum_rewd

            self.avg_rewd.append( self.sum_rewd )
            if len( self.avg_rewd ) > avg_length : self.avg_rewd.popleft()

            now = time.time()
            self.elapsed_time = ( now - self.timer )
            self.timer = now

            self.acc_time += self.elapsed_time

            print( '*** Episode : %5d | Time : %6.2f s | Rewards : %9.3f | Average : %9.3f |' % \
                   ( episode + 1 , self.elapsed_time , self.sum_rewd ,
                     sum( self.avg_rewd ) / len( self.avg_rewd ) ) , end = '' )
            #self.env.info()

            self.episode_rewards.append(self.sum_rewd)
            self.episode_lengths.append(self.elapsed_time)
            self.time_rewards.append([self.sum_rewd, self.acc_time])
            self.accumulated_lenghts.append(self.acc_time)

            plot_episode_stats(self.episode_lengths, self.episode_rewards, self.accumulated_lenghts, self.time_rewards)

            self.sum_rewd = 0
