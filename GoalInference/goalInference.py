from rrtPlanner import rrtPlanner
from obstacleMap import obstacleMap
import obstacleMap as omap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin
import numpy as np
from networkx import nx
import sys as sys
from multiprocessing import Pool
from timeit import default_timer as timer
import copy

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
	path = nx.shortest_path(rrt.graph, source=rrtPlanner.STARTID, target=rrtPlanner.GOALID)
	path_edges = list(zip(path,path[1:]))
else:
	print("Could not be solved")
	sys.exit()

# Start a trajectory along this path
# Take a point on some proportion of this path
intermediary   = int(len(path)/2)
obs_path_edges = list(zip(path[0:intermediary],path[1:intermediary]))

# Plot
plt.figure(1, figsize=(8, 8))
plt.axis('equal')
plt.imshow(obsmap.pixels)
nx.draw_networkx_nodes(rrt.graph, rrt.graph.pos, node_size=10, node_color='b', alpha=0.25)
nx.draw_networkx_edges(rrt.graph, rrt.graph.pos, alpha=0.4)
nx.draw_networkx_nodes(rrt.graph, rrt.graph.pos,nodelist=path,node_size=10,node_color='r', alpha=0.25)
nx.draw_networkx_edges(rrt.graph, rrt.graph.pos,edgelist=path_edges,edge_color='r',width=1)
nx.draw_networkx_edges(rrt.graph, rrt.graph.pos,edgelist=obs_path_edges,edge_color='g',width=1)

fig = plt.figure(2, figsize=(8, 8))
plt.axis('equal')
axes = plt.imshow(obsmap.pixels)
plt.show()


def inner_func():
	all_spath=[]
	print("Launch ")
	for i in range(20):
		print(i)
		lrrt = rrtPlanner(obsmap)
    	# New goals
    	# Select a free, random goal
		xg   = np.random.randint(0,width_-1)
		yg   = np.random.randint(0,height_-1)
		while (omap.testCollision(obsmap.pixels,xg,yg)):
			xg = np.random.randint(0,width_-1)
			yg = np.random.randint(0,height_-1)
		lrrt.SetGoal(xg,yg)
		# Solve the planning problem
		paths      = lrrt.Solve()
		path       = []
		path_edges = []
		# Get the resulting shortest path (if it does exist)
		if paths>0:
			path       = nx.shortest_path(lrrt.graph, source=rrtPlanner.STARTID, target=rrtPlanner.GOALID)
			path_edges = list(zip(path,path[1:]))
			all_spath.append(path_edges)
	return all_spath
			


pool = Pool(processes=10) 
results = [pool.apply(inner_func, args=()) for x in range(1,5)]
results = [y for x in results for y in x] 

print(results)
print(len(results))
mttime = timer()-start
print("MP took %f seconds" % mttime)

start = timer()
inner_func()
singletime = timer()-start
print("Single took %f seconds" % singletime)



plt.show()

