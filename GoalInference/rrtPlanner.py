import numpy as np
import math as math
from obstacleMap import obstacleMap
from networkx import nx
from numba import jit

# Nearest node
@jit(nopython=True)
def Nearest(pos,norms,numNodes,zRand):  
    idNearest= 1 
    dmin     = norms[idNearest] - 2.0 * np.dot(pos[idNearest], zRand)
    for idNode in range(1,numNodes):
        zNode = pos[idNode]
        d     = norms[idNode] - 2.0 * np.dot(zNode, zRand)
        if  d[0] < dmin[0]:
            idNearest = idNode
            dmin      = norms[idNode] - 2.0 * np.dot(zNode, zRand)
    return idNearest            

# Tree expansion
@jit(nopython=True)
def StepFromTo(nFrom,zFrom,zTo,epsilon):
    d = nFrom - 2.0 * np.dot(zFrom, zTo) + np.dot(zTo, zTo)
    if d[0] < epsilon:
        return zTo
    else:
        theta = math.atan2(zTo[1]-zFrom[1],zTo[0]-zFrom[0])
        return np.array([zFrom[0] + epsilon*math.cos(theta), zFrom[1] + epsilon*math.sin(theta)])

class rrtPlanner(): 
    MAXITERATIONS= 4000 
    MAXNODES     = 10000
    GOALID       = 0
    STARTID      = 1
    #RRT: Inspired from S. Lavalle's code
    #http://msl.cs.uiuc.edu/~lavalle/sub/rrt.py
    def __init__(self, map_, epsilon_=10.0, tolerance_=10.0):
        # Class variables
        self.epsilon    = epsilon_
        self.tolerance  = tolerance_
        self.endCounter = 0
        self.graph      = nx.Graph()
        self.graph.add_node(0) # Goal  node
        self.graph.add_node(1) # Start node
        self.graph.pos     = np.zeros((rrtPlanner.MAXNODES,2),dtype=float)  # locations
        self.graph.norms   = np.zeros((rrtPlanner.MAXNODES,1),dtype=float)  # norms
        self.obstacleMap   = map_

    # Goal initialization
    def SetGoal(self,xg_,yg_):        
        # Draw goal
        self.xg = xg_
        self.yg = yg_
        self.graph.pos[rrtPlanner.GOALID]  = np.array([xg_,yg_])  
        self.graph.norms[rrtPlanner.GOALID]= np.dot([xg_,yg_], [xg_,yg_])  

    # Reset graph and goal
    def Reset(self):
        self.graph        = nx.Graph()
        self.graph.add_node(0)
        self.graph.add_node(1)
        self.graph.pos     = np.zeros((rrtPlanner.MAXNODES,2),dtype=float)  # locations
        self.graph.norms   = np.zeros((rrtPlanner.MAXNODES,1),dtype=float)  # norms
        self.endCounter    = 0
        self.graph.remove_nodes_from(self.graph[0])

    # Solve
    def Solve(self):
        for i in range(rrtPlanner.MAXITERATIONS):
            if self.endCounter==10:
                return self.endCounter
            # Sample a 2D point    
            zRand               = np.multiply(np.random.rand(2),np.array([self.obstacleMap.width,self.obstacleMap.height]).reshape(2))
            idNearest           = Nearest(self.graph.pos,self.graph.norms,len(self.graph.nodes),zRand) 
            zNearest            = self.graph.pos[idNearest] 
            zNew                = StepFromTo(self.graph.norms[idNearest],zNearest,zRand,self.epsilon)
            if (zNew[0]<0):
                continue
            if (zNew[1]<0):
                continue
            if (zNew[0]>=self.obstacleMap.width):
                continue
            if (zNew[1]>=self.obstacleMap.height):
                continue
    
            # Test collision
            if (self.obstacleMap.pixels[int(zNew[1]),int(zNew[0])]==obstacleMap.OBSCOLOR).all():
                continue    
            else:    
                idNew = len(self.graph.nodes)
                self.graph.add_node(idNew)
                self.graph.pos[idNew]   = zNew  
                self.graph.norms[idNew] = np.dot(zNew,zNew)  
                self.graph.add_edge(idNearest,idNew)
                # Test if goal is reached
                if abs(zNew[1]-self.yg)<self.tolerance and abs(zNew[0]-self.xg)<self.tolerance:
                    self.graph.add_edge(idNew, rrtPlanner.GOALID)
                    self.endCounter = self.endCounter + 1
        return self.endCounter              
              