
import copy
import numpy as np

### Get Batch Data (Numpy)
def get_batch_numpy( data , b , i ):

    l = len( data )

    st = ( i + 0 ) * b % l
    fn = ( i + 1 ) * b % l

    if st > fn :
        if fn == 0: return data [ st : l ]
        else: return np.vstack( ( data[ st : l ] , data[ 0 : fn ] ) )
    else:
        return data[ st : fn ]

### Get Batch Data (List)
def get_batch_list( data , b , i ):

    l = len( data )

    st = ( i + 0 ) * b % l
    fn = ( i + 1 ) * b % l

    if st > fn :
        if fn == 0: return data [ st : l ]
        else: return data[ st : l ] + data[ 0 : fn ]
    else:
        return data[ st : fn ]

### Get Batch Data
def get_batch( data , b , i ):

    if isinstance( data , list ):
        return get_batch_list( data , b , i )

    if isinstance( data , np.ndarray ):
        return get_batch_numpy( data , b , i )

    print( 'DATA TYPE NOT SUPPORTED' )
    return None

### Get Data Seqlen
def get_data_seqlen( data ):

    data_seqlen = []

    max_seqlen = 0
    for i in range( len( data ) ):

        seqlen = len( data[i] )
        data_seqlen.append( seqlen )

        if seqlen > max_seqlen:
            max_seqlen = seqlen

    return data_seqlen

### Pad Data
def pad_data( data , max_seqlen = None ):

    data_seqlen = get_data_seqlen( data )
    if max_seqlen is None: max_seqlen = max( data_seqlen )

    max_featlen = len( data[0][0] )
    pad = [ 0.0 for _ in range( max_featlen ) ]

    for i in range( len( data ) ):
        data[i] += [ pad for _ in range( max_seqlen - data_seqlen[i] ) ]

    return data_seqlen


