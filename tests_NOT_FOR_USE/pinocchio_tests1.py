import pinocchio as pin
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper

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

URDF = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_px100_pin_ver.urdf"
robot = pin.buildModelFromUrdf(URDF)
print("model name: " + robot.name)

data = robot.createData()

q = pin.randomConfiguration(robot)
print("q: %s" % q.T)

pin.forwardKinematics(robot, data, q)

for name, oMi in zip(robot.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}".format( name, *oMi.translation.T.flat )))
