from sources.source_vrep import source_vrep
import numpy as np
import time

##### SOURCE VREP POPPY
class source_vrep_poppy( source_vrep ):

    RENDER = False
    DESIRED_POSITION = [ 30, 45, -20, 90, 45, -45 ]


    ### __INIT__
    def __init__( self ):

        source_vrep.__init__( self , 'poppy_ergo_jr' )

        # Get objects
        self.objects = []
        for i in range(6):
            obj = self.env.get_object_by_name('m{}'.format(i + 1), is_joint=True)
            self.objects.append(obj)

        # Start episode
        self.counter = 0

    ### INFORMATION
    def num_actions( self ): return 6
    def range_actions( self ): return 1

    # STEP ACTIONS
    def step(self, positions, speeds=None):

        for i, m in enumerate(self.objects):
            target = positions[i]
            if i == 0:
                target *= -1
            m.set_position_target(target)

            if speeds is not None:
                m.set_velocity(speeds[i])

    # GET OBSERVATIONS
    def _get_obsv(self, desired_position):

        self.counter += 1

        # Observations
        out_pos = np.zeros(6, dtype=np.float32)
        out_vel = np.zeros(6, dtype=np.float32)
        out_dis = np.zeros(6, dtype=np.float32)
        for i, m in enumerate(self.objects):
            angle = m.get_joint_angle()
            out_pos[i] = angle
            out_vel[i] = m.get_joint_velocity()[0]
            out_dis[i] = desired_position[i] - angle
        #obsv = np.append(out_pos,out_dis)
        obsv = out_pos

        # Rewards
        reward = 0
        for i, m in enumerate(self.objects):
            dist_abs = 180 - np.abs( m.get_joint_angle() - desired_position[i] ) #* (i + 1)
            reward += dist_abs / 10000

        # Dones
        done = True
        for i, m in enumerate(self.objects):
            done *= ( np.square( m.get_joint_angle() - desired_position[i] ) <= 0.5 )
        if self.counter == 200:
            self.counter = 0
            done = True
            self.env.stop_simulation()
            time.sleep(.2)

        if (self.counter % 200 == 0): print(obsv)

        return obsv, reward, done

    ### SCALE AND LIMIT ACTIONS
    def map_keys( self , actn ):

        return np.multiply(actn, 90)
