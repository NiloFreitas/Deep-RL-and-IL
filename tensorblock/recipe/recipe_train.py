
import numpy as np
import tensorblock as tb

class recipe_train:

####### Train
    def train( self , **args ):

        pars = { **self.defs_train , **args }

        if pars['train_length'] is not None:
            if pars['train_data'] is not None: pars['train_data'] = pars['train_data'][ :pars['train_length'] , ]
            if pars['train_labels'] is not None: pars['train_labels'] = pars['train_labels'][ :pars['train_length'] , ]

        if pars['test_length'] is not None:
            if pars['test_data'] is not None: pars['test_data'] = pars['test_data'][ :pars['test_length'] , ]
            if pars['test_labels'] is not None: pars['test_labels'] = pars['test_labels'][ :pars['test_length'] , ]

        if not isinstance( pars['eval_function'] , list ):
            pars['eval_function'] = [ pars['eval_function'] ]

        flag_eval = pars['eval_function'] is not None
        flag_plot = pars['plot_function'] is not None

        flag_save = pars['saver'] is not None
        flag_summary = pars['summary'] is not None and \
                       pars['writer'] is not None

        print( '######################################################################## TRAINING' )

        self.train_evaluate( pars , flag_summary )
        if flag_plot:
            pars_plot = self.pars( pars['plot_function'] )
            tb.plotters.initialize( pars_plot['shape'] )
            self.train_plot( pars , pars_plot )

        if isinstance( pars['train_data'] , list       ): num_samples = len( pars['train_data'] )
        if isinstance( pars['train_data'] , np.ndarray ): num_samples = pars['train_data'].shape[0]

        num_batches = int( num_samples / pars['size_batch'] ) + 1

        for epoch in range( pars['num_epochs'] ):

            for batch in range( num_batches ):
                self.train_optimize( pars , batch )

            if flag_eval and ( epoch + 1 ) % pars['eval_freq'] == 0:
                self.train_evaluate( pars , flag_summary , epoch )

            if flag_plot and ( epoch + 1 ) % pars['plot_freq'] == 0:
                self.train_plot( pars , pars_plot , epoch )

            if flag_save and ( epoch + 1 ) % pars['save_freq'] == 0 :
                self.train_save( pars )

        print( '######################################################################## END TRAINING' )

####### Optimize
    def train_optimize( self , pars , batch ):

        train_dict = self.prepare( pars['train_data'] , pars['train_labels'] , pars['train_seqlen'] ,
                                   pars['size_batch'] , batch )
        self.run( pars['optimizer'] , train_dict , use_dropout = True )

####### Evaluate
    def train_evaluate( self , pars , flag_summary , epoch = -1 ):

        test_dict = self.prepare( pars['test_data'] , pars['test_labels'] , pars['test_seqlen'] )
        eval = self.run( pars['eval_function'] , test_dict , use_dropout = False )

        if flag_summary:

            summ = self.run( pars['summary'] , test_dict , use_dropout = False )
            self.write( name = pars['writer'] , summary = summ , iter = epoch + 1 )

        print( '*** Epoch' , epoch + 1 , '| ' , end = '' )
        for i , function in enumerate( pars['eval_function'] ):
            print( function + ' :' , eval[i] , '| ' , end = '' )
        print()

####### Plot
    def train_plot( self , pars , pars_plot , epoch = -1 ):

        x = pars['test_data'][ : np.prod( pars_plot['shape'] ) , : ]
        y = self.run( 'Output' , [ [ 'Input' , x ] ] , use_dropout = False )

        self.tensor( pars['plot_function'] )( x , y , epoch = epoch + 1 ,
                dir = pars_plot['dir'] , shape = pars_plot['shape'] )

####### Save
    def train_save( self , pars ):

        self.save( name = pars['saver'] )

####### Prepare Data
    def prepare( self , data , labels , seqlen , size_batch = None , batch = None ):

        dict = []

        if data is not None:
            if size_batch is None: batch_data = data
            else: batch_data = tb.aux.get_batch( data , size_batch , batch )
            dict.append( [ 'Input' , batch_data ] )
        if labels is not None:
            if size_batch is None: batch_labels = labels
            else: batch_labels = tb.aux.get_batch( labels , size_batch , batch )
            dict.append( [ 'Label' , batch_labels ] )
        if seqlen is not None:
            if size_batch is None: batch_seqlen = seqlen
            else: batch_seqlen = tb.aux.get_batch( seqlen , size_batch , batch )
            dict.append( [ 'SeqLen' , batch_seqlen ] )

        return dict


