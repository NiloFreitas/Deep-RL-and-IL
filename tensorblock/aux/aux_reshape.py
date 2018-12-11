
import math
import numpy as np

### Flat Dimension
def flat_dim( shape ):

    return np.prod( shape[1:] )

### Flatten
def flatten( x ):

    return x.reshape( [ -1 , flat_dim( x ) ] )

### 2D Side Dimension
def side2D( x , d = 1 ):

    if isinstance( x , list ) : x = x[1]
    return int( round( math.pow( x / d , 1.0 / 2.0 ) ) )

### 3D Side Dimension
def side3D( x , d = 1 ):

    if isinstance( x , list ) : x = x[1]
    return int( round( math.pow( x / d , 1.0 / 3.0 ) ) )

### Shape 1D
def shape1D( x ):

    return [ None , x.shape[1] ]

### Shape 2D
def shape2D( x , channels = 1 ):

    side = side2D( x.shape[1] , channels )
    return [ None , side , side , channels ]

### Shape 3D
def shape3D( x , channels = 1 ):

    side = side3D( x.shape[1] , channels )
    return [ None , side , side , side , channels ]

### Spread
def spread( x , n ):

    return x if isinstance( x , list ) else [ x ] * n

### Flatten List
def flatten_list( list1 ):

    list2 = []

    for item1 in list1:
        if isinstance( item1 , list ):
            for item2 in item1:
                list2.append( item2 )
        else:
            list2.append( item1 )

    return list2

