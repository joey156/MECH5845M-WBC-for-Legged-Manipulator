import pinocchio as pin
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np

R = eye(3); p = zero(3)
M0 = pin.SE3(R, p)
M = pin.SE3.Random()
M.translation = p; M.rotation = R

v = zero(3); w = zero(3)
nu0 = pin.Motion(v, w)
nu = pin.Motion.Random()
nu.linear = v; nu.angular = w

f = zero(3); tau = zero(3)
phi0 = pin.Force(f, tau)
phi = pin.Force.Random()
phi.linear = f; phi.angular = tau

URDF = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_wx200.urdf"

robot, collision_model, visual_model = pin.buildModelsFromUrdf(URDF)
print("model name: " + robot.name)

data, collision_data, visual_data = pin.createDatas(robot, collision_model, visual_model)
print(robot.nq)
print(robot.nv)
q = pin.neutral(robot)
print(type(q))
print("q: %s" % q.T)
print(q)
joints = "0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
print(joints)
joints_split = joints.split(",")
print(joints_split)
joints_float = []
for i in range(len(joints_split)):
    joints_float.append(float(joints_split[i]))

print(joints_float)
q2 = np.array(joints_float)
print(q2)

pin.forwardKinematics(robot, data, q2)

pin.updateGeometryPlacements(robot, data, collision_model, collision_data)

pin.updateGeometryPlacements(robot, data, visual_model, visual_data)

print("\nJoint placements: ")
for name, oMi in zip(robot.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)))

print("\nCollision object placements: ")
for k, oMg in enumerate(collision_data.oMg):
    print(("{:d} : {: .2f} {: .2f} {: .2f}".format(k, *oMg.translation.T.flat)))


