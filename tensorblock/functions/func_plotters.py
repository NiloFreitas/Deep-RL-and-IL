
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import clf , plot , draw , show

import tensorblock as tb

### Plot Initialize
def initialize( shape ):

    plt.figure( figsize = ( 12 , 9 ) )

### Plot Reconstruction
def reconst( x1 , x2 ,
             epoch = 0 , dir = 'figures' , shape = None ):

    if len( x1.shape ) == 2:
        s = tb.aux.side2D( x1.shape[1] ) ; x1 = x1.reshape( [ -1 , s , s , 1 ] )
    if len( x2.shape ) == 2:
        s = tb.aux.side2D( x2.shape[1] ) ; x2 = x2.reshape( [ -1 , s , s , 1 ] )

    r , c = shape ; k = 0
    for j in range( r ):
        for i in range( c ):

            plt.subplot( 2 * r , c , i + 2 * j * c + 1 )
            plt.imshow( x1[ k , : , : , 0 ] , vmin = 0 , vmax = 1 )
            plt.axis( 'off' )

            plt.subplot( 2 * r , c , i + 2 * j * c + c + 1 )
            plt.imshow( x2[ k , : , : , 0 ] , vmin = 0 , vmax = 1 )
            plt.axis( 'off' )

            k = k + 1

    if not os.path.exists( dir ): os.makedirs( dir )
    plt.savefig( dir + '/epoch%d.png' % epoch , bbox_inches = 'tight' )

