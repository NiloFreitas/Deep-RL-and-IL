
import os
import tensorflow as tf

class recipe_save:

####### Add Saver
    def addSaver( self , name = None , input = None , dir = 'models' , pref = 'model' ):

        name = self.add_label(
                self.savers , 'Saver' , name , add_order = False )

        if input is None :
            input = self.folder

        collection = self.collection( input )
        self.savers.append( [ tf.train.Saver( collection ) , [ name , dir , pref ] ] )

####### Save
    def save( self , name = None , dir = None , pref = None , iter = None ):

        if name is None: name = self.savers[-1][1][0]
        if dir  is None: dir  = self.pars( name )[1]
        if pref is None: pref = self.pars( name )[2]
        if dir[-1] is not '/': dir += '/'

        if not os.path.exists( dir ):
            os.makedirs( dir )

        self.tensor( name ).save( self.sess ,
                global_step = iter , save_path = dir + pref )

####### Restore
    def restore( self , name = None , dir = None , pref = None ):

        if name is None: name = self.savers[-1][1][0]
        if dir  is None: dir  = self.pars( name )[1]
        if pref is None: pref = self.pars( name )[2]
        if dir[-1] is not '/': dir += '/'

        if os.path.exists( dir ):
            self.tensor( name ).restore( self.sess , save_path = dir + pref )
