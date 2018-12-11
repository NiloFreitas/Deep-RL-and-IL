
import numpy as np

### Parse Pais
def parse_pairs( data ):

    if not isinstance( data , list ):
        return [ [ data , None ] ]

    for i in range( len( data ) ):
        if not isinstance( data[i] , list ):
            data[i] = [ data[i] , None ]

    return data

### Get Folder
def get_folder( name ):

    folder , list = '' , name.split( '/' )
    for i in range( len( list ) - 1 ):
        folder += list[i] + '/'
    return folder

### Clean Duplicates
def clean_dups( str ):

    clean = ''
    for i in range( 1 , len( str ) ):
        if str[i] is not str[i-1]:
            clean += str[i]

    return clean

