import pickle
import numpy as np
import tensorflow as tf
import tensorblock as tb
import cv2

### Create Dataset
def create_dataset( tensors, extras, pars ):

    s_dataset = tf.data.Dataset.list_files(pars['data_path'])

    def _s_parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels = 3)
        image_decoded = tf.py_func( lambda x : cv2.resize( x , ( pars['width'] , pars['height'] ) ), [image_decoded], tf.uint8 )
        image_decoded = tf.py_func( lambda x : cv2.cvtColor( x, cv2.COLOR_RGB2GRAY ), [image_decoded], tf.uint8 )
        return image_decoded

    s_dataset = s_dataset.map(_s_parse_function)
    s_dataset = s_dataset.batch(pars['b_size'])
    s_dataset = s_dataset.repeat()

    a_dataset = tf.data.TextLineDataset(pars['label_path'])
    a_dataset = a_dataset.batch(pars['b_size'])
    a_dataset = a_dataset.repeat()

    iterator1 = s_dataset.make_one_shot_iterator()
    image = iterator1.get_next()

    iterator2 = a_dataset.make_one_shot_iterator()
    label = iterator2.get_next()

    return image, label

### Load Matrix
def load_mat( file ):

    lines = [ line.rstrip('\n') for line in open( file ) ]

    nd = lines[0].split( ' ' )
    mat = np.zeros( ( int(nd[0]) , int(nd[1]) ) )

    for i in range( 2 , len( lines ) ):
        list = lines[i].split( ' ' )
        for j in range( 0 , len( list ) - 1 ):
            mat[i-2,j] = float( list[j] )

    return mat

### Save List
def save_list( file , list ):

    pickle.dump( list , open( file + '.lst' , 'wb' ) )

### Load List
def load_list( file ):

    return pickle.load( open( file + '.lst' , 'rb' ) )

### Load Numpy
def load_numpy( file ):

    return np.load( file + '.npy' )
