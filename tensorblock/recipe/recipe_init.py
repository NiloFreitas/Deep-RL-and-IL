
import tensorflow as tf
import tensorblock as tb

class recipe_init:

####### Initialize Variables
    def initVariables( self ):

        self.labels = {}

        self.cnt = 0
        self.curr_input = None

        self.blocks , self.order = [] , []
        self.layers , self.extras = [] , []

        self.inputs , self.variables = [] , []
        self.weights , self.biases , self.dropouts = [] , [] , []
        self.operations = []

        self.summaries , self.writers = [] , []
        self.savers , self.plotters = [] , []

####### Initialize Defaults
    def initDefaults( self ):

        self.defs_block = {
            'src' : None , 'dst' : None , 'type' : None ,
            'mod_inputs' : True , 'mod_variables' : True , 'mod_layers' : True ,
            'no_ops' : False ,
        }

        self.defs_input = {
            'name' : None , 'shape' : None , 'tensor' : None ,
            'out_sides' : None , 'out_channels' : None ,
            'copy' : None , 'share' : None , 'first_none' : True ,
            'dtype' : tf.float32 ,
        }

        self.defs_variable = {
            'name' : None , 'shape' : None , 'tensor' : None ,
            'out_sides' : None , 'out_channels' : None ,
            'first_none' : False ,

            'type' : tb.vars.truncated_normal ,
            'copy' : None , 'share' : None ,
            'mean' : 0.0  , 'stddev' : 0.1 ,
            'value' : 0.0 , 'min' : 0.0 , 'max' : 1.0 ,
            'trainable' : True , 'seed' : None ,
        }

        self.defs_operation = {
            'name' : None , 'function' : None ,
            'input' : None , 'extra' : None , 'src' : None , 'dst' : None ,
            'learning_rate' : 1e-4 ,
        }

        self.defs_train = {
            'train_data' : None , 'train_labels' : None , 'train_seqlen' : None , 'train_length' : None ,
            'test_data'  : None  , 'test_labels' : None , 'test_seqlen'  : None , 'test_length'  : None ,
            'size_batch' : 100 , 'num_epochs' : 10 ,

            'optimizer' : None ,
            'summary' : None , 'writer' : None ,
            'saver' : None , 'save_freq' : 10 ,

            'eval_function' : None , 'eval_freq' : 1 ,
            'plot_function' : None , 'plot_freq' : 1 ,
        }

        self.defs_plotter = {
            'name' : None , 'function' : None ,
            'dir' : 'figures' , 'shape' : [ 2 , 5 ] ,
        }

        self.defs_layer = {
            'input' : None , 'type' : None , 'name' : None ,
            'copy' : None , 'share' : None , 'label' : None ,

            'weight_type' : tb.vars.truncated_normal ,
            'weight_name' : None , 'weight_copy' : None , 'weight_share' : None ,
            'weight_mean' : 0.0  , 'weight_stddev' : 0.1 ,
            'weight_value' : 0.0 , 'weight_min' : 0.0 , 'weight_max' : 1.0 ,
            'weight_trainable' : True , 'weight_seed' : None ,

            'bias_type' : tb.vars.truncated_normal ,
            'bias_name' : None , 'bias_copy' : None , 'bias_share' : None ,
            'bias_mean' : 0.0 , 'bias_stddev' : 0.1 ,
            'bias_value' : 0.0 , 'bias_min' : 0.0 , 'bias_max' : 1.0 ,
            'bias_trainable' : True , 'bias_seed' : None ,

            'dropout_name' : None , 'dropout' : 0.0 ,
            'dropout_copy' : None , 'dropout_share' : None ,

            'in_sides'    : None , 'out_sides'    : None ,
            'in_channels' : None , 'out_channels' : None ,

            'pooling' : 1 , 'pooling_ksize' : None ,
            'pooling_strides' : None , 'pooling_padding' : None ,

            'cell_type' : 'LSTM' , 'num_cells' : None ,
            'in_dropout'  : 0.0 , 'in_dropout_name'  : None ,
            'out_dropout' : 0.0 , 'out_dropout_name' : None ,
            'seqlen' : None ,

            'strides' : 1 , 'ksize' : 3 , 'padding' : 'SAME' ,
            'activation' : tb.activs.relu , 'activation_pars' : None
        }

####### Set Input Defaults
    def setInputDefaults( self , **args ):
        self.defs_input = { **self.defs_input , **args }

####### Set Layer Defaults
    def setLayerDefaults( self , **args ):
        self.defs_layer = { **self.defs_layer , **args }

####### Set Operation Defaults
    def setOperationDefaults( self , **args ):
        self.defs_operation = { **self.defs_operation , **args }

####### Set Variable Defaults
    def setVariableDefaults( self , **args ):
        self.defs_variable = { **self.defs_variable , **args }

####### Initialize
    def initialize( self , vars = None ):

        if vars is None : vars = self.folder

        collection = self.collection( vars )
        self.sess.run( tf.variables_initializer( collection ) )
