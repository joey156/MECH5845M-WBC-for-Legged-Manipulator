import pybullet as p
import time
import pinocchio as pin
import numpy as np
from QP_Wrapper import QP
from Robot_Wrapper import RobotModel

# setup simulation parameters
p.connect(p.GUI)
plane = p.loadURDF("/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/tests_NOT_FOR_USE/plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
urdfFlags = p.URDF_USE_SELF_COLLISION
urdf_path = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_wx200.urdf"

# load the legged robot urdf
LeggedRobot_bullet = p.loadURDF(urdf_path,[0,0,0.4],[0,0,0,1], flags=urdfFlags, useFixedBase=False)

# Removing the collision pairs between the two gripper fingers and the bar they are attached to
p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 28, 29, 0)
p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 26, 28, 0)
p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 26, 29, 0)

# initialise the RobotModel class
LeggedRobot = RobotModel(urdf_path)

# Initialising lists
jointIds = []
paramIds = []
joints_py = []
joints_pin = []
joints_pin_base = [0,0,0,0,0,0,0]

# Enables a trackbar to select the max joint force
maxForceId = p.addUserDebugParameter("maxForce",0,100,100)

# This loop sets the linear and angular damping to 0 and adds all actuated joints to jointIds
for j in range(p.getNumJoints(LeggedRobot_bullet)):
    p.changeDynamics(LeggedRobot_bullet, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(LeggedRobot_bullet, j)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            jointIds.append(j)

# Simulation camera settings
p.getCameraImage(480,320)
p.setRealTimeSimulation(0)

# setting objectives
com_target_pos = np.array([[0.004, 0.001, -0.003]]).T #base: 0.004, 0.001, -0.003

planner_pos, planner_vel = LeggedRobot.posAndVelTargetsCoM(com_target_pos)



EE_pos_FL = np.array([[0.163, 0.144, -0.326]]).T
EE_pos_FR = np.array([[0.174, -0.142, -0.329]]).T
EE_pos_RL = np.array([[-0.203, 0.148, -0.319]]).T
EE_pos_RR = np.array([[-0.196, -0.148, -0.32]]).T
EE_pos_GRIP = np.array([[0.252, 0., 0.207]]).T
EE_vel = np.array([[0,0,0,0,0,0]]).T
EE_target_pos = [EE_pos_FL, EE_pos_FR, EE_pos_RL, EE_pos_RR, EE_pos_GRIP]
EE_target_vel = [EE_vel, EE_vel, EE_vel, EE_vel, EE_vel]

# Main while loop
while (1):
    # Initial configuration
    maxForce = p.readUserDebugParameter(maxForceId)
    joints_py = LeggedRobot.current_joint_config[7:]
    print(joints_py)
    for i in range(len(joints_py)):
        p.setJointMotorControl2(LeggedRobot_bullet, jointIds[i], p.POSITION_CONTROL, joints_py[i], force=maxForce)
    p.stepSimulation()
    time.sleep(1./500)
    t = time.time()
    while (time.time()- t) < 2:
            p.stepSimulation()
            time.sleep(1./500)
    
    for i in range(len(planner_pos)):
        maxForce = p.readUserDebugParameter(maxForceId)

        # Fetch and combine joint position and velocity limits
        lower_vel_lim, upper_vel_lim = LeggedRobot.jointVelLimitsArray()
        lower_pos_lim, upper_pos_lim = LeggedRobot.jointPosLimitsArray()
        lb = np.concatenate((lower_vel_lim.T, lower_pos_lim), axis=0).reshape((52,))
        ub = np.concatenate((upper_vel_lim.T, upper_pos_lim), axis=0).reshape((52,))
        #lb = lower_pos_lim.reshape((26,))
        #ub = upper_pos_lim.reshape((26,))
        #lb = lower_vel_lim.reshape((26,))
        #ub = upper_vel_lim.reshape((26,))
        
        #print(planner_pos[i])
        # Fetch the new A and b for QP
        A = LeggedRobot.qpCartesianA()
        b = LeggedRobot.qpCartesianB(planner_pos[i], planner_vel[i], EE_target_pos, EE_target_vel).reshape((33,))
        # Solve QP
        qp = QP(A, b, lb, ub)
        q_vel = qp.solveQP()

        # Find the new joint angles and update the robot model (pinocchio model)
        LeggedRobot.jointVelocitiestoConfig(q_vel, True)

        # Update the joint configurations
        joints_py = LeggedRobot.current_joint_config[7:]
        for ii in range(len(joints_py)):
            p.setJointMotorControl2(LeggedRobot_bullet, jointIds[ii], p.POSITION_CONTROL, joints_py[ii], force=maxForce)

        p.stepSimulation()
        time.sleep(1./500)
        #print(i)

    while (1):
        print("done")
        print(LeggedRobot.robot_data.com[0])
        print(joints_py)
        p.stepSimulation()
        time.sleep(10)


# Cleanup
p.disconnect()