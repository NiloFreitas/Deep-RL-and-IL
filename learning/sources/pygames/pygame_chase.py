
import pygame
import pygame.surfarray as surfarray
import random

import time
import numpy as np

class pygame_chase:

    ### START SIMULATION
    def reset( self ):

        pygame.init()

        self.black = [  0  ,  0  ,  0  ]
        self.white = [ 255 , 255 , 255 ]
        self.blue  = [  0  ,  0  , 255 ]
        self.red   = [ 255 ,  0  ,  0  ]
        self.green = [  0  , 255 ,  0  ]

        self.screen_size = [ 640 , 480 ]
        self.border_pos = [ 20 , 20 ]

        self.n_enemies =  5
        self.n_friends = 10

        self.circ_dia = 20
        self.circ_rad = int( self.circ_dia / 2 )
        self.circ_rad2 = self.circ_rad ** 2

        self.hero_pos = [ self.screen_size[0] / 2 , self.screen_size[1] / 2 ]
        self.hero_spd , self.hero_acl = [ 0 , 0 ] , 3

        self.res , self.dst = 10 , 100
        self.angles = list( np.arange( 0 , 360 , self.res ) * np.pi / 180.0 )
        self.coses = [ np.cos( ang ) for ang in self.angles ]
        self.sines = [ np.sin( ang ) for ang in self.angles ]

        self.left = self.border_pos[0]
        self.right = self.screen_size[0] - self.border_pos[0] - self.circ_dia

        self.top = self.border_pos[1]
        self.bottom = self.screen_size[1] - self.border_pos[1] - self.circ_dia

        self.spd_min , self.spd_max = -5 , 5

        self.screen = pygame.display.set_mode( self.screen_size , 0 , 32 )

        self.border_size = [ self.screen_size[0] - 2 * self.border_pos[0] ,
                             self.screen_size[1] - 2 * self.border_pos[1] ]

        background_surf = pygame.Surface( self.screen_size )
        self.background = background_surf.convert()
        self.background.fill( self.black )
        self.background.fill( self.white , ( self.border_pos[0]  , self.border_pos[1]  ,
                                             self.border_size[0] , self.border_size[1] ) )


        hero_surf = pygame.Surface( [ self.circ_dia , self.circ_dia ] )
        pygame.draw.circle( hero_surf , self.blue , [ self.circ_rad , self.circ_rad ] , self.circ_rad )
        self.hero = hero_surf.convert() ; self.hero.set_colorkey( self.black )

        friend_surf = pygame.Surface( [ self.circ_dia , self.circ_dia ] )
        pygame.draw.circle( friend_surf , self.green , [ self.circ_rad , self.circ_rad ] , self.circ_rad )
        self.friend = friend_surf.convert() ; self.friend.set_colorkey( self.black )

        enemy_surf = pygame.Surface( [ self.circ_dia , self.circ_dia ] )
        pygame.draw.circle( enemy_surf , self.red , [ self.circ_rad , self.circ_rad ] , self.circ_rad )
        self.enemy = enemy_surf.convert() ; self.enemy.set_colorkey( self.black )

        self.friends = []
        for _ in range( self.n_friends ):
            self.friends.append( self.create() )

        self.enemies = []
        for _ in range( self.n_enemies ):
            self.enemies.append( self.create() )

        self.catch , self.miss = 0 , 0

        return self.draw()

    ### INFO ON SCREEN
    def info( self ):

        print( ' Score : %3d x %3d |' % \
                 ( self.catch , self.miss ) , end = '' )

    ### DRAW SCREEN
    def draw( self ):

        self.screen.blit( self.background, ( 0 , 0 ) )

        for friend in self.friends:
            self.screen.blit( self.friend , friend[0] )
        for enemy in self.enemies:
            self.screen.blit( self.enemy , enemy[0] )

        ret_pos = [ 0 , 0 ]
        self.sensor_pos = [ self.hero_pos[0] + self.circ_rad , self.hero_pos[1] + self.circ_rad ]
        f_poss , e_poss = self.possibles()

        observation = []

        for i in range( len( self.angles ) ):

            hit = 0
            for d in range( 10 , self.dst , 5 ):

                ret_pos[0] = self.hero_pos[0] + d * self.coses[i]
                ret_pos[1] = self.hero_pos[1] + d * self.sines[i]

                hit = self.visible( ret_pos , f_poss , e_poss )
                if hit > 0: break

            ret_pos[0] += self.circ_rad
            ret_pos[1] += self.circ_rad

            color = self.blue if hit == 0 else self.green if hit == 1 else self.red if hit == 2 else self.black
            pygame.draw.line( self.screen , color , self.sensor_pos , ret_pos , 2 )

            if hit == 0: observation.append( self.dst )
            else: observation.append( self.distance( ret_pos , self.sensor_pos ) )
            observation.append( hit )

        self.screen.blit( self.hero , self.hero_pos )
        pygame.display.update()

        return np.array( observation )

    ### MOVE ONE STEP
    def step( self , action ):

        # Execute Action

        self.hero_spd[0] *= 0.5
        self.hero_spd[1] *= 0.5

        if action == 1: self.hero_spd[0] += self.hero_acl
        if action == 2: self.hero_spd[0] -= self.hero_acl
        if action == 3: self.hero_spd[1] += self.hero_acl
        if action == 4: self.hero_spd[1] -= self.hero_acl

        # Propagate

        self.hero_pos[0] += self.hero_spd[0]
        self.hero_pos[1] += self.hero_spd[1]

        self.bounce( [ self.hero_pos , self.hero_spd ] )

        for friend in self.friends:
            self.move( friend )
            self.bounce( friend )

        for enemy in self.enemies:
            self.move( enemy )
            self.bounce( enemy )

        # Create Observation

        obsv = self.draw()

        # Collect Rewards and Check Done

        rewd = self.collect( obsv )
        done = self.catch >= 10 or self.miss >= 10

        # Return Data

        return obsv , rewd , done

############################################################################################

    ### Create Ball
    def create( self ):

        pos = [ random.randint( self.left + 2 * self.circ_rad , self.right  - 2 * self.circ_rad ) ,
                random.randint( self.top  + 2 * self.circ_rad , self.bottom - 2 * self.circ_rad ) ]

        spd = [ random.randint( self.spd_min , self.spd_max ) ,
                random.randint( self.spd_min , self.spd_max ) ]

        if spd[0] == 0: spd[0] = +1
        if spd[1] == 0: spd[1] = -1

        return [ pos , spd ]

    ### Move Ball
    def move( self , posspd ):

        posspd[0][0] += posspd[1][0]
        posspd[0][1] += posspd[1][1]

    ### Bounce Ball
    def bounce( self , posspd ):

        pos , spd = posspd

        if pos[0] < self.left:   pos[0] , spd[0] = 2 * self.left   - pos[0] , - spd[0]
        if pos[0] > self.right:  pos[0] , spd[0] = 2 * self.right  - pos[0] , - spd[0]
        if pos[1] < self.top:    pos[1] , spd[1] = 2 * self.top    - pos[1] , - spd[1]
        if pos[1] > self.bottom: pos[1] , spd[1] = 2 * self.bottom - pos[1] , - spd[1]

    ### Collide
    def collide( self , pos ):

        return self.distance( self.hero_pos , pos ) < self.circ_dia

    ### Distance
    def distance( self , pos1 , pos2 ):

        return np.sqrt( ( pos1[0] - pos2[0] )**2 +
                        ( pos1[1] - pos2[1] )**2 )

    ### Possibles
    def possibles( self ):

        friends_possibles = []
        for i , friend in enumerate( self.friends ):
            if self.distance( friend[0] , self.hero_pos ) < self.dst:
                friends_possibles.append( i )

        enemies_possibles = []
        for i , enemy in enumerate( self.enemies ):
            if self.distance( enemy[0] , self.hero_pos ) < self.dst:
                enemies_possibles.append( i )

        return friends_possibles , enemies_possibles

    ### Visible
    def visible( self , pos , f_poss , e_poss ):

        if pos[0] + self.circ_rad < self.left or pos[0] - self.circ_rad > self.right or \
           pos[1] + self.circ_rad < self.top  or pos[1] - self.circ_rad > self.bottom:
            return 3

        for i in f_poss:
            if self.distance( self.friends[i][0] , pos ) < self.circ_rad:
                return 1

        for i in e_poss:
            if self.distance( self.enemies[i][0] , pos ) < self.circ_rad:
                return 2

        return 0

    ### Collect
    def collect( self , obsv ):

        reward = 0

        for i , friend in enumerate( self.friends ):
            if self.collide( friend[0] ): # If Collided with Friend
                self.friends[i] = self.create()
                reward -= 1 ; self.miss += 1

        for i , enemy in enumerate( self.enemies ):
            if self.collide( enemy[0] ): # If Collided with Enemy
                self.enemies[i] = self.create()
                reward += 1 ; self.catch += 1

        for i in range( 0 , obsv.shape[0] , 2 ):
            if obsv[ i + 1 ] == 3 : # If Near a Wall
                reward -= ( 1.0 - obsv[ i ] / self.dst ) / len( self.angles )

        return reward
