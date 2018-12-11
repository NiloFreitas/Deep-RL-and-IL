
import pygame
import pygame.surfarray as surfarray
import random

class pygame_catch:

    ### START SIMULATION
    def reset( self ):

        pygame.init()

        self.black = [  0  ,  0  ,  0  ]
        self.white = [ 255 , 255 , 255 ]

        self.screen_size = [ 320 , 240 ]
        self.bar_size    = [   6 ,  30 ]

        self.bar_pos = [ 10 , self.screen_size[1] / 2 - self.bar_size[1] / 2 ]
        self.bar_spd , self.bullet_spd = 7.0 , [ 8.0 , 0.0 ]

        self.max_bullets , self.prob_bullets = 1 , 0.1
        self.wait_frames , self.count_frames = 25 , 0

        self.bullet_diam = 8
        self.bullet_rad = int( self.bullet_diam / 2 )
        self.bullets = []

        self.screen = pygame.display.set_mode( self.screen_size , 0 , 32 )

        background_surf = pygame.Surface( self.screen_size )
        self.background = background_surf.convert()
        self.background.fill( self.black )

        self.bar = pygame.Surface( self.bar_size ).convert()
        self.bar.fill( self.white )

        bullet_surf = pygame.Surface( [ self.bullet_diam , self.bullet_diam ] )
        pygame.draw.circle( bullet_surf , self.white , [ self.bullet_rad , self.bullet_rad ] , self.bullet_rad )

        self.bullet = bullet_surf.convert()
        self.bullet.set_colorkey( self.black )

        self.bar_max_top = 0
        self.bar_max_bot = self.screen_size[1] - self.bar_size[1]

        self.catch , self.miss = 0 , 0

        return self.draw()

    ### INFO ON SCREEN
    def info( self ):

        print( ' Score : %3d x %3d |' % \
                 ( self.catch , self.miss ) , end = '' )

    ### DRAW SCREEN
    def draw( self ):

        self.screen.blit( self.background, ( 0 , 0 ) )

        self.screen.blit( self.bar , self.bar_pos )
        for bullet in self.bullets:
            self.screen.blit( self.bullet , bullet[0] )

        pygame.display.update()
        return pygame.surfarray.array3d( pygame.display.get_surface() )

    ### MOVE ONE STEP
    def step( self , action ):

        # Initialize Reward

        rewd = 0.0

        # Execute Action

        if action == 0 : self.bar_pos[1] -= self.bar_spd
        if action == 2 : self.bar_pos[1] += self.bar_spd

        # Restrict Up and Down Motion

        if self.bar_pos[1] < self.bar_max_top: self.bar_pos[1] = self.bar_max_top
        if self.bar_pos[1] > self.bar_max_bot: self.bar_pos[1] = self.bar_max_bot

        # Move Bullets and Check for Collision

        remove = None
        for i , bullet in enumerate( self.bullets ):

            bullet[0][0] -= bullet[1]
            if bullet[0][0] < self.bar_pos[0] + self.bar_size[0]:
                if bullet[0][1] + self.bullet_rad > self.bar_pos[1] and \
                   bullet[0][1] + self.bullet_rad < self.bar_pos[1] + self.bar_size[1]:
                    self.catch += 1 ; rewd += + 1.0 # Positive Reward
                else:
                    self.miss += 1 ; rewd += - 1.0 # Negative Reward
                remove = i

        # Remove Bullet

        if remove is not None:
            self.bullets.pop( remove )

        # Add Bullets

        if self.count_frames == 0:
            if len( self.bullets ) < self.max_bullets:
                if random.random() < self.prob_bullets or len( self.bullets ) == 0:
                    self.count_frames = self.wait_frames
                    self.bullets.append( [
                        [ self.screen_size[0] - 10 ,
                          self.bar_size[1] / 2 + ( self.screen_size[1] - self.bar_size[1] ) * random.random() ] ,
                        self.bullet_spd[0] + self.bullet_spd[1] * random.random() ] )
        else:
            self.count_frames -= 1

        # Determine Done

        done = self.catch == 25 or self.miss == 25

        # Return Data

        return self.draw() , rewd , done

############################################################################################
