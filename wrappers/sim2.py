import pybullet as p
import time
import pinocchio as pin
import numpy as np
from QP_Wrapper import QP
from Robot_Wrapper import RobotModel
from klampt.model import trajectory

# setup simulation parameters
p.connect(p.GUI)
plane = p.loadURDF("/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/tests_NOT_FOR_USE/plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
urdfFlags = p.URDF_USE_SELF_COLLISION
urdf_path = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_wx200.urdf"
#urdf_path = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_px100_pin_ver.urdf"
# load the legged robot urdf
LeggedRobot_bullet = p.loadURDF(urdf_path,[0,0,0.43],[0,0,0,1], flags=urdfFlags, useFixedBase=False)

# Removing the collision pairs between the two gripper fingers and the bar they are attached to

p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 28, 29, 0)
p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 26, 28, 0)
p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 26, 29, 0)
"""
p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 27, 28, 0)
p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 25, 27, 0)
p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 25, 28, 0)
"""
# initialise the RobotModel class
EE_frame_names = ["FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed", "gripper_bar"]
EE_joint_names = ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint", "gripper"]
#LeggedRobot = RobotModel(urdf_path, "FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed", "gripper_bar", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint", "gripper", "waist", "imu_joint")
LeggedRobot = RobotModel(urdf_path, EE_frame_names, EE_joint_names, "waist", "imu_joint", "FR_hip_joint")
print(LeggedRobot.current_joint_config)

# Initialising lists
jointIds = []
paramIds = []
joints_py = []
joints_pin = []
joints_pin_base = [0,0,0,0,0,0,0]
joints_py_current = []

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
p.getCameraImage(1000,1000)
#p.resetDebugVisualizerCamera(1.0,1.25,-19.8,[0.07,0.1,0.07])
p.resetDebugVisualizerCamera(1.00,63.65,-31.4,[0.04,0.03,0.13])
p.setRealTimeSimulation(0)

# select the tasks that are active
LeggedRobot.setTasks(EE=True, CoM=False, Trunk=True)

# setting objectives
EE_pos_FL = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[0]].translation]).T #np.array([[0.174, -0.142, -0.32]]).T
EE_pos_FR = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[1]].translation]).T #np.array([[0.163, 0.144, -0.32]]).T
EE_pos_RL = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[2]].translation]).T #np.array([[-0.196, -0.148, -0.32]]).T
EE_pos_RR = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[3]].translation]).T #np.array([[-0.203, 0.148, -0.32]]).T
EE_pos_GRIP = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[4]].translation]).T #np.array([[0.202, 0., 0.227]]).T #base: 0.252 0. 0.207 reach 0.5 0 -0.15
EE_vel = np.array([[0,0,0,0,0,0]]).T
Trunk_target_pos = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.trunk_frame_index].translation]).T # 0.013 0.002 0.001
Trunk_target_vel = np.array([[0,0,0,0,0,0]]).T
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

    t = time.time()

    while (time.time()- t) < 2:
            p.stepSimulation()
            time.sleep(1./500)

    # reset current joint configuration list
    joints_py_current = []
        
    # get curret joint configuration for feedback loop
    for i in range(len(jointIds)):
        joints_py_current.append(p.getJointStates(LeggedRobot_bullet, jointIds)[i][0])

    
    
    # convert list into array
    joints_py_feedback = np.array(joints_py_current)
    #print(joints_current_py)

    # fetch the IMU data for potision and orientation in the world frame
    imu_state = p.getLinkState(LeggedRobot_bullet, 0)
    base_config = np.concatenate((LeggedRobot.current_joint_config[:3], np.array(imu_state[5])), axis=0)
    print(base_config)
    #LeggedRobot.updateState(joints_py, base_config)

    # get CoM position
    base_pos = imu_state[0]
    previouse_com_pos = np.add(LeggedRobot.robot_data.com[0], base_pos).tolist()

    com_target_pos = np.array([np.add(LeggedRobot.robot_data.com[0], np.array([0., 0., 0]))]).T #base: 0.005, 0.001, -0.002

    print(com_target_pos)

    planner_pos, planner_vel = LeggedRobot.posAndVelTargetsCoM(com_target_pos)

    # setting gripper trajectory
    current_gripper_pos = LeggedRobot.robot_data.oMf[53].translation
    print(current_gripper_pos)
    gripper_displacement = np.array([0.4, 0.02, -0.15])
    milestones = [current_gripper_pos.tolist(), [0.4, 0., 0.24], [0.4, -0.3, 0.24], [-0.1, -0.3, 0.24], [-0.1, -0.3, -0.01], [0.4, -0.3, -0.01], [0.4, 0.3, -0.01], [-0.1, 0.3, -0.01], [-0.1, 0.3, 0.24],[0.4, 0.3, 0.24], [0.4, 0, 0.24]]
    #milestones = [current_gripper_pos.tolist(), [0.25, 0., 0.21], [0.25, -0.18, 0.21], [-0.06, -0.18, 0.21], [-0.06, -0.18, 0.1], [0.25, -0.18, 0.1], [0.25, 0.18, 0.1], [-0.06, 0.18, 0.1], [-0.06, 0.18, 0.21],[0.25, 0.18, 0.21], [0.25, 0, 0.21]]
    #milestones = [current_gripper_pos.tolist(), np.add(current_gripper_pos, np.array([0.1, 0, 0])).tolist()]
    traj = trajectory.Trajectory(milestones=milestones)
    traj2 = trajectory.HermiteTrajectory()
    traj2.makeSpline(traj)  
    traj_interval = np.arange(0,len(milestones),0.0005).tolist()

    # plot desired trajectory
    previouse_traj_point = np.add(traj2.eval(0), base_pos).tolist()
    for i in np.arange(0,len(milestones),0.1).tolist():
        current_traj_point = np.add(traj2.eval(i), base_pos).tolist()
        p.addUserDebugLine(previouse_traj_point, current_traj_point, lineColorRGB=[0,0,0], lineWidth=3, lifeTime=0)
        previouse_traj_point = current_traj_point

    print(LeggedRobot.com_weight)
    
    t = time.time()
    while (time.time()- t) < 2:
            p.stepSimulation()
            time.sleep(1./500)
    print("start")
    for i in traj_interval:

        #print(planner_pos[i])
        
        maxForce = p.readUserDebugParameter(maxForceId)

        # Find new gripper position
        EE_pos_GRIP = np.array([traj2.eval(i)]).T
        EE_target_pos[4] = EE_pos_GRIP

        # fetch the IMU data for potision and orientation in the world frame
        imu_state = p.getLinkState(LeggedRobot_bullet, 0)
        base_config = np.concatenate((LeggedRobot.current_joint_config[:3], np.array(imu_state[5])), axis=0)

        #st = time.time()
        # run the WBC to solve for the new joint configurations
        joint_config = LeggedRobot.runWBC(base_config, target_cartesian_pos_CoM=planner_pos[0], target_cartesian_vel_CoM=planner_vel[0], target_cartesian_pos_EE=EE_target_pos, target_cartesian_vel_EE=EE_target_vel, target_cartesian_pos_trunk=Trunk_target_pos, target_cartesian_vel_trunk=Trunk_target_vel)
        #print(time.time()-st)

        # update simulation with new joint configurations
        for ii in range(len(joints_py)):
            p.setJointMotorControl2(LeggedRobot_bullet, jointIds[ii], p.POSITION_CONTROL, joint_config[ii], force=maxForce)

        # visually track the CoM
        base_pos = imu_state[0]
        current_com_pos = np.add(LeggedRobot.robot_data.com[0], base_pos).tolist()

        #p.addUserDebugLine(previouse_com_pos, current_com_pos, lineColorRGB=[255,0,0], lineWidth=10, lifeTime=0, parentObjectUniqueId=100)

        previouse_com_pos = current_com_pos

        p.stepSimulation()
        time.sleep(1./500)

        #print(i)

    while (1):
        print("done")
        print("CoM:")
        print(LeggedRobot.robot_data.com[0])
        print("\n")
        print("FL:")
        print(LeggedRobot.robot_data.oMf[19].translation)
        print("\n")
        print("FR:")
        print(LeggedRobot.robot_data.oMf[11].translation)
        print("\n")
        print("RL:")
        print(LeggedRobot.robot_data.oMf[27].translation)
        print("\n")
        print("RR:")
        print(LeggedRobot.robot_data.oMf[35].translation)
        print("\n")
        print("Grip:")
        print(LeggedRobot.robot_data.oMf[53].translation)
        print(joints_py)
        p.stepSimulation()
        break
    break


# Cleanup
p.disconnect()
