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


# Test if a segment intersects the obstacle map
@jit(nopython=True)
def TestCollisionSegment(posA,posB,pmap): 
    for s in range(1,100):
        posNew = (s/100.0)*posA+(1.0-(s/100.0))*posB
        # Test collision
        # TODO: replace the value of 100 by a constant
        if pmap[int(posNew[1]),int(posNew[0])][0]==100:
            return 1
    return 0            

# Refinement of a shortest path
@jit(nopython=True)
def Refine(pos,path,pmap): 
    for i in range(1,100):
        l = len(path)
        # Select one point randomly on the path (but not the last one)
        k = np.random.randint(0,l-1)
        # Select a length
        d = np.random.randint(2,5)
        if d+k>l-1:
            continue
        # Check if it is possible to shorten the path
        posA = pos[path[k]]
        posB = pos[path[k+d]]
        if TestCollisionSegment(posA,posB,pmap)==1:
            continue
        del path[k+1:k+d]    


# Evaluate the squared distance from a point pt to a segment
def SquaredDistanceToSegment(pt,pt1,pt2):
    # Orthogonal projection of pt on the line pt1,pt2
    d12sq   = math.pow(pt1[0]-pt2[0],2.0)+math.pow(pt1[1]-pt2[1],2.0)
    orthdir = np.asarray((pt2[1]-pt1[1],-(pt2[0]-pt1[0])))/math.sqrt(d12sq)
    proj    = pt+(orthdir[0]*(pt1[0]-pt[0])+orthdir[1]*(pt1[1]-pt[1]))*orthdir    
    if ((proj[1]-pt1[1])*(pt2[1]-pt1[1])+(proj[0]-pt1[0])*(pt2[0]-pt1[0])<0.0):
        return (pt[1]-pt1[1])*(pt[1]-pt1[1])+(pt[0]-pt1[0])*(pt[0]-pt1[0])
    else:
        if ((proj[1]-pt1[1])*(pt2[1]-pt1[1])+(proj[0]-pt1[0])*(pt2[0]-pt1[0])<d12sq):
            return (pt[1]-proj[1])*(pt[1]-proj[1])+(pt[0]-proj[0])*(pt[0]-proj[0])
        else:
            return (pt[1]-pt2[1])*(pt[1]-pt2[1])+(pt[0]-pt2[0])*(pt[0]-pt2[0])    

# Evaluate the squared distance from a point pt to a path given as an array of points
def SquaredDistanceToPath(pt,path):
    rows,cols = path.shape
    dmin=pow(10.0,20.0)
    for i in range(0,rows-1):
        di=SquaredDistanceToSegment(pt,path[i,:],path[i+1,:])
        if di<dmin:
            dmin = di
    return dmin        

# Generate a sub-path
def GenerateObservedPath(pos,path):
    # Compute total length
    l = len(path)
    t = 0.0
    for i in range(0,l-1):
        t = t + math.sqrt(math.pow(pos[path[i+1]][0]-pos[path[i]][0],2.0)+math.pow(pos[path[i+1]][1]-pos[path[i]][1],2.0))
    tobs = np.random.uniform(0.5*t,t)     
    t = 0.0
    obspath = []       
    for i in range(0,l-1):
        obspath.append(pos[path[i]])
        d = math.sqrt(math.pow(pos[path[i+1]][0]-pos[path[i]][0],2.0)+math.pow(pos[path[i+1]][1]-pos[path[i]][1],2.0))
        if t + d>tobs:
            lastp = pos[path[i]]+(tobs-t)/d*(pos[path[i+1]]-pos[path[i]])
            obspath.append(lastp)
            break    
        t = t + d    
    return np.asarray(obspath)   


# Shortest path        
def ShortestPath(graph,source,target,omap):
    path         = nx.shortest_path(graph, source, target)
    Refine(graph.pos,path,omap.pixels)
    return path

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
            if self.obstacleMap.pixels[int(zNew[1]),int(zNew[0])][0]==obstacleMap.OBSCOLOR[0]:
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
              