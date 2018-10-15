from rrtPlanner import rrtPlanner
import rrtPlanner as rrtp

from obstacleMap import obstacleMap
import obstacleMap as omap

import matplotlib
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin
import numpy as np
from networkx import nx
import sys as sys
from multiprocessing import Pool
from timeit import default_timer as timer
import copy
import math as math

start = timer()

width_, height_ = 700, 700

# Create the map
obsmap = obstacleMap(width_, height_)

# Create the rrt planner
rrt    = rrtPlanner(obsmap)

# Select a free, random goal
xg = np.random.randint(0,width_-1)
yg = np.random.randint(0,height_-1)
while (omap.testCollision(obsmap.pixels,xg,yg)):
        xg = np.random.randint(0,width_-1)
        yg = np.random.randint(0,height_-1)
rrt.SetGoal(xg,yg)

# Solve by generating a RRT that is stopped when close to the goal
paths      = rrt.Solve()
path       = []
path_edges = []

if paths>0:
	path         = rrtp.ShortestPath(rrt.graph,rrtPlanner.STARTID,rrtPlanner.GOALID,obsmap)
	path_edges = list(zip(path,path[1:]))
else:
	print("Problem could not be solved")
	sys.exit()

# Start a trajectory along this path
# Take a point on some proportion of this path
obs_path_edges = rrtp.GenerateObservedPath(rrt.graph.pos,path)
print(obs_path_edges)

# Plot
plt.figure(1, figsize=(8, 8))
plt.axis('equal')
plt.imshow(obsmap.pixels)
nx.draw_networkx_nodes(rrt.graph, rrt.graph.pos, node_size=10, node_color='b', alpha=0.25)
nx.draw_networkx_edges(rrt.graph, rrt.graph.pos, alpha    =0.4)
nx.draw_networkx_nodes(rrt.graph, rrt.graph.pos, nodelist =path,node_size=10,node_color='r', alpha=0.25)
nx.draw_networkx_edges(rrt.graph, rrt.graph.pos, edgelist =path_edges,edge_color='r',width=1)
plt.plot(obs_path_edges[:,0],obs_path_edges[:,1],color='g',linewidth=2)

fig = plt.figure(2, figsize=(8, 8))
plt.axis('equal')
axes = plt.imshow(obsmap.pixels)


def multipleSolver(id):
	all_spath=[]
	print("Launch process %d"%id)
	for i in range(50):
		lrrt = rrtPlanner(obsmap)
    	# New goals
    	# Select a free, random goal
    	# TODO: improve the proposal as m(z'_G;z)
		xg   = np.random.randint(0,width_-1)
		yg   = np.random.randint(0,height_-1)
		while (omap.testCollision(obsmap.pixels,xg,yg)):
			xg = np.random.randint(0,width_-1)
			yg = np.random.randint(0,height_-1)
		lrrt.SetGoal(xg,yg)
		# Solve the planning problem
		# The 
		paths      = lrrt.Solve()
		path       = []
		path_edges = []
		# Get the resulting shortest path (if it does exist)
		if paths>0:
			#path         = nx.shortest_path(lrrt.graph, source=rrtPlanner.STARTID, target=rrtPlanner.GOALID)
			path         = rrtp.ShortestPath(lrrt.graph,rrtPlanner.STARTID,rrtPlanner.GOALID,obsmap)
			path_xy      = np.asarray([lrrt.graph.pos[x] for x in path])
			all_spath.append(path_xy)
	return all_spath
			


pool    = Pool(processes=6) 
results = [pool.apply_async(multipleSolver, args=[x]) for x in range(1,5)]
results = [x for res in results for x in res.get(timeout=1500) ]

mttime = timer()-start
print("Multi processing took %f seconds" % mttime)

# Show all the shortest paths found
for p in results:
	plt.plot(p[:,0],p[:,1],color='k')
nx.draw_networkx_edges(rrt.graph, rrt.graph.pos, edgelist =path_edges,edge_color='r',width=2)

# For all the points of the observed path, project them on the segment (0,0)-(100,0)
plt.plot(obs_path_edges[:,0],obs_path_edges[:,1],color='g',linewidth=3)

# For each sampled path, evaluate the likelihood of the partial observation. 
Z            = results[0]
oldlikelihood= 0.00000001
for p in results:
	Zp   = p 	
	# Evaluate the likelihood for this Z'
	sumd = 0.0
	sump = 0.0
	for pt in obs_path_edges[:]:
		sumd+= rrtp.SquaredDistanceToPath(pt,p)
		sump+= 1.0
	sqdist        = (sumd/sump)	
	likelihood    = math.exp(-sqdist/100.0)

	# And decide to accept the move or not (MCMC) otherwise keep the old sample
	tau  = np.random.uniform(0,1)   
	alpha= likelihood/oldlikelihood  
	if alpha>=tau:
			Z = Zp
			plt.scatter(Z[-1,0],Z[-1,1],s=100.0*likelihood,color='k',alpha=0.5)	
		
plt.show()

