
import tensorflow as tf
import tensorblock as tb

class recipe_summary:

####### Add Summary Base
    def addSummaryBase( self , inputs , name , function ):

        inputs = tb.aux.parse_pairs( inputs )

        name = self.add_label(
                self.summaries , 'Summary' , name , add_order = False )

        list = []
        for input in inputs:
            if input[1] is None: input[1] = self.folder + input[0]
            tensor , tag = self.tensor( input[0] ) , input[1]
            list.append( function( tag , tensor ) )
        self.summaries.append( [ list , name ] )

####### Add Summary Scalar
    def addSummaryScalar( self , input , name = None ):
        return self.addSummaryBase( input , name , tf.summary.scalar )

####### Add Summary Histogram
    def addSummaryHistogram( self , input , name = None ):
        return self.addSummaryBase( input , name , tf.summary.histogram )

####### Add Summary
    def addSummary( self , input = None , name = None ):

        name = self.add_label(
                self.summaries , 'Summary' , name , add_order = False )

        if input is None: input = tf.summary.merge_all()
        else: input = tf.merge_summary( self.tensor_list( input ) )

        self.summaries.append( [ input , name ] )

####### Add Writer
    def addWriter( self , name = None , dir = 'logs' ):

        name = self.add_label(
                self.writers , 'Writer' , name , add_order = False )

        self.writers.append(
            [ tf.summary.FileWriter( dir , graph = self.sess.graph ) , name ] )

####### Write
    def write( self , name = None , summary = None , iter = None ):

        if name is None: name = self.writers[-1][1]
        self.tensor( name ).add_summary( summary , iter )
