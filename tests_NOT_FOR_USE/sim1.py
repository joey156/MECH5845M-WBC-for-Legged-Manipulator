import pybullet as p
import time
import pinocchio as pin
import numpy as np

# setup simulation parameters:
p.connect(p.GUI)
plane = p.loadURDF("plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
urdfFlags = p.URDF_USE_SELF_COLLISION
urdf_path = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_wx200.urdf"

# Load the Legged robot URDF
LeggedRobot_py = p.loadURDF(urdf_path,[0,0,0.4],[0,0,0,1],flags = urdfFlags,useFixedBase=False)

# Load the Legged robot URDF to a pinocchio model and create data
LeggedRobot_pin = pin.buildModelFromUrdf(urdf_path)
LeggedRobot_data = LeggedRobot_pin.createData()

# Display joint info
for j in range (p.getNumJoints(LeggedRobot_py)):
        print(p.getJointInfo(LeggedRobot_py,j))

# Removing the collision pairs between the two gripper fingers and the bar they are attached to
p.setCollisionFilterPair(LeggedRobot_py, LeggedRobot_py, 28, 29, 0)
p.setCollisionFilterPair(LeggedRobot_py, LeggedRobot_py, 26, 28, 0)
p.setCollisionFilterPair(LeggedRobot_py, LeggedRobot_py, 26, 29, 0)

# Initialising lists
jointIds = []
paramIds = []
joints_py = []
joints_pin = []
joints_pin_base = [0,0,0,0,0,0,0]

# Enables a trackbar to select the max joint force
maxForceId = p.addUserDebugParameter("maxForce",0,100,50)
print(p.getNumJoints(LeggedRobot_py))
# This loop sets the linear and angular damping to 0 and adds all actuated joints to jointIds
for j in range(p.getNumJoints(LeggedRobot_py)):
    p.changeDynamics(LeggedRobot_py, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(LeggedRobot_py, j)
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
    with open("test1.txt", "r") as filestream:
        for line in filestream:
            maxForce = p.readUserDebugParameter(maxForceId)
            currentline = line.split(",")
            frame = currentline[0]
            t = currentline[1]
            joints_py = currentline[2:]
            for i in range(len(joints_py)):
                joints_pin_base.append(float(joints_py[i]))
            joints_pin = np.array(joints_pin_base)
            #print(joints_pin)
            for j in range(len(joints_py)):
                targetPos = float(joints_py[j])
                states = p.getJointState(LeggedRobot_py, jointIds[j])
                #print("JointID: " + str(jointIds[j]) + "  jointPosition: " + str(states[0]),"  jointVelocity: ", str(states[1]))
                p.setJointMotorControl2(LeggedRobot_py, jointIds[j], p.POSITION_CONTROL, targetPos, force=maxForce)
            
            IMU_state = p.getLinkState(LeggedRobot_py, 1)
            pin.forwardKinematics(LeggedRobot_pin, LeggedRobot_data, joints_pin)
            p.stepSimulation()
            time.sleep(1./500.)
            #print("\n Joint positions: \n")
            #for name, oMi in zip(LeggedRobot_pin.names, LeggedRobot_data.oMi):
                #print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)))
            # Fetch the configuration of the floating base
            joints_pin_base = []
            for i in range(2):
                for ii in range(len(IMU_state[i])):
                    joints_pin_base.append(IMU_state[i][ii])
            #print(IMU_state[:2])
            #print(joints_pin_base)
        


# Cleanup
p.disconnect()
