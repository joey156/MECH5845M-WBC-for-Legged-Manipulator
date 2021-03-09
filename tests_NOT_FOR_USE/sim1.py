import pybullet as p
import time

# setup simulation parameters:
p.connect(p.GUI)
plane = p.loadURDF("plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
urdfFlags = p.URDF_USE_SELF_COLLISION

# Load the Legged robot URDF
LeggedRobot = p.loadURDF("/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/pin_test.urdf",[0,0,0.4],[0,0,0,1],flags = urdfFlags,useFixedBase=False)

# Display joint info
for j in range (p.getNumJoints(LeggedRobot)):
        print(p.getJointInfo(LeggedRobot,j))

# Removing the collision pairs between the two gripper fingers and the bar they are attached to
p.setCollisionFilterPair(LeggedRobot, LeggedRobot, 27, 28, 0)
p.setCollisionFilterPair(LeggedRobot, LeggedRobot, 25, 27, 0)
p.setCollisionFilterPair(LeggedRobot, LeggedRobot, 25, 28, 0)

# Initialising lists
jointIds = []
paramIds = []
joints = []

# Enables a trackbar to select the max joint force
maxForceId = p.addUserDebugParameter("maxForce",0,100,50)

# This loop sets the linear and angular damping to 0 and adds all actuated joints to jointIds
for j in range(p.getNumJoints(LeggedRobot)):
    p.changeDynamics(LeggedRobot, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(LeggedRobot, j)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
        jointIds.append(j)

print(jointIds)

# Simulation camera settings
p.getCameraImage(480,320)
p.setRealTimeSimulation(0)

# Main while loop
while (1):
    with open("mocap_with_arm.txt", "r") as filestream:
        for line in filestream:
            maxForce = p.readUserDebugParameter(maxForceId)
            currentline = line.split(",")
            frame = currentline[0]
            t = currentline[1]
            joints = currentline[2:]
            for j in range(len(joints)):
                targetPos = float(joints[j])
                p.setJointMotorControl2(LeggedRobot, jointIds[j], p.POSITION_CONTROL, targetPos, force=maxForce)

            p.stepSimulation()
            time.sleep(1./500.)


# Cleanup
p.disconnect()
