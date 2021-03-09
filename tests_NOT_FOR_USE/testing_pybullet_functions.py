import pybullet as p
import time

# setup simulation parameters:
p.connect(p.GUI)
plane = p.loadURDF("plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)

# Load the Legged robot URDF
LeggedRobot = p.loadURDF("/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/pin_test.urdf",[0,0,0.4],[0,0,0,1])

# get number of in info for the joints
numJoints = p.getNumJoints(LeggedRobot)
print(numJoints)
infoJoints_base = p.getJointInfo(LeggedRobot, 0)
print(infoJoints_base)
infoJoints_FR_hip_joint = p.getJointInfo(LeggedRobot, 2)
print(infoJoints_FR_hip_joint)
statesJoints_base = p.getJointState(LeggedRobot, 0)
print(statesJoints_base)
statesJoints_FR_hip_joint = p.getJointState(LeggedRobot, 2)
print(statesJoints_FR_hip_joint)
