from klampt.model import trajectory
from klampt import vis
import numpy as np

milestones = [[0.252,0,0.207],[0.35,0,0.207],[0.5,0,0],[0.5,0,-0.15]]

traj = trajectory.Trajectory(milestones=milestones)


vis.add("point",[0,0,0])
vis.animate("point",traj)
vis.add("traj",traj)
vis.spin(float('inf'))   #show the window until you close it

traj2 = trajectory.HermiteTrajectory()
traj2.makeSpline(traj)

#vis.animate("point",traj2)
#vis.add("traj2",traj2)
#vis.spin(float('inf'))

#print(np.array(traj2.eval(0.1)).shape)

