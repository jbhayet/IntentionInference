import numpy as np


#@jit(nopython=True)
def testCollision(omap,x,y):                 
        return (omap[x,y]==obstacleMap.OBSCOLOR).all()

class obstacleMap():
    OBSCOLOR     = np.asarray([100,100,100])
    GOALCOLOR    = np.asarray([200,100,100])
    FREECOLOR    = np.asarray([255,255,255])   

    def __init__(self, width_, height_,nobstacles_=10):
        # Class variables
        self.nobstacles = nobstacles_
        self.pixels     = 255*np.ones((width_, height_, 3), dtype=np.uint8)
        self.width      = width_
        self.height     = height_
        # Generate obstacles
        for i in range(self.nobstacles):
            x = np.random.random_integers(0,self.width)
            y = np.random.random_integers(0,self.height)
            w = 20+np.random.random_integers(0,self.width/2)
            h = 20+np.random.random_integers(0,self.height/2)
            for k in range(x,x+w):
                for l in range(y,y+h): 
                    if k<self.width and l<self.height:
                        self.pixels[l,k]=obstacleMap.OBSCOLOR



