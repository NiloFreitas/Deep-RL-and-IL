
import tensorflow as tf

class recipe_plot:

####### Add Plotter
    def addPlotter( self , **args ):

        pars = { **self.defs_plotter , **args }
        pars['name'] = self.add_label(
                self.plotters , 'Plotter' , pars['name'] , add_order = False )

        self.plotters.append( [ pars['function'] , pars ] )

