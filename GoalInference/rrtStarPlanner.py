import numpy as np
import math as math
from networkx import nx
from rrtPlanner import rrtPlanner


class rrtStarPlanner(rrtPlanner):
    def __init__(self, pixels_, width_, height_, epsilon_=10.0, numNodes_= 5000, nobstacles_=10):
        rrtPlanner.__init__(self, pixels_, width_, height_, epsilon_, numNodes_, nobstacles_)

    # Solve
    def Solve(self):
        for i in range(rrtPlanner.MAXITERATIONS):
            if self.terminated:
                return self.goalId 
            # Sample a 2D point    
            rand = np.random.random_integers(0,self.width), np.random.random_integers(0,self.height)
            #if np.random.random_integers(0,100)<3: 
            #    rand=(self.xg,self.yg)
            rr2= np.dot(rand, rand)
            nn = self.graph.pos[0]
            dmin= math.sqrt(np.dot(nn, nn) - 2 * np.dot(nn, rand) + rr2)
            n  = 0
            for p in nx.nodes(self.graph):
                if p!=self.goalId:
                    pp = self.graph.pos[p]
                    if math.sqrt(np.dot(pp, pp) - 2 * np.dot(pp, rand) + rr2) < dmin:
                        nn  = pp
                        n   = p
                        dmin= math.sqrt(np.dot(pp, pp) - 2 * np.dot(pp, rand) + rr2)
            newnode = self.step_from_to(nn,rand)

            # Test collision
            if (self.pixels[newnode[1],newnode[0]]==rrtPlanner.OBSCOLOR).all():
                continue    
            else:    
                # Test if goal is reached
                if (self.pixels[newnode[1],newnode[0]]==rrtPlanner.GOALCOLOR).all():
                    self.terminated = True  
                l = len(self.graph.pos)
                self.graph.add_node(l)
                self.graph.pos[l] = newnode  
                self.graph.add_edge(n, l)
                if self.terminated:
                    self.graph.add_edge(l, self.goalId)
        return -1            
              