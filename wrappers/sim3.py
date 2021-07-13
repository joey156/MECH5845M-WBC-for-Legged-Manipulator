import pybullet as p
import time
import pinocchio as pin
import numpy as np
import pandas as pd
from QP_Wrapper import QP
from Robot_Wrapper4 import RobotModel
from klampt.model import trajectory

# define robot to use: "a1_wx200", "a1_px100", "laikago_vx300"
robot = "a1_wx200"
#robot = "a1_px100"
#robot = "laikago_vx300"
if robot == "a1_wx200":
    robot_index = 0
    foot_offset = True
if robot == "a1_px100":
    robot_index = 1
    foot_offset = True
if robot == "laikago_vx300":
    robot_index = 2
    foot_offset = False

# select simulation
sim = "static_reach"
#sim = "chickenhead"
if sim == "static_reach":
    sim_id = 0
if sim == "chickenhead":
    sim_id = 1

# enable or disable data collection
#data_collection = True
data_collection = False
if data_collection == True:
    data_dict = {"Time":[], "target x":[], "target y":[], "target z":[], "real x":[], "real y":[], "real z":[]}
    data_name = robot + "_" + sim + ".csv"


# setup simulation parameters
p.connect(p.GUI)
plane = p.loadURDF("/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/tests_NOT_FOR_USE/plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
urdfFlags = p.URDF_USE_SELF_COLLISION

# defining urdf file paths
a1_wx200_urdf_path = "/home/joey156/Documents/Robot_Descriptions/urdf/a1_wx200.urdf"
a1_px100_urdf_path = "/home/joey156/Documents/Robot_Descriptions/urdf/a1_px100_pin_ver.urdf"
laikago_vx300_urdf_path = "/home/joey156/Documents/Robot_Descriptions/urdf/laikago_vx300.urdf"
urdf_path_list = [a1_wx200_urdf_path, a1_px100_urdf_path, laikago_vx300_urdf_path]

mesh_dir_path = "/home/joey156/Documents/Robot_Descriptions/meshes"

# initialise the RobotModel class
EE_frame_names = ["FR_foot_fixed", "FL_foot_fixed", "RR_foot_fixed", "RL_foot_fixed", "gripper_bar"]
EE_joint_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint", "gripper"]
hip_waist_joint_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint", "waist"]
#LeggedRobot = RobotModel(urdf_path, "FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed", "gripper_bar", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint", "gripper", "waist", "imu_joint")
LeggedRobot = RobotModel(urdf_path_list[robot_index], mesh_dir_path, EE_frame_names, EE_joint_names, "waist", "imu_joint", "FR_hip_joint", hip_waist_joint_names, foot_offset=foot_offset)
print(LeggedRobot.current_joint_config)

# load the legged robot urdf
LeggedRobot_bullet = p.loadURDF(urdf_path_list[robot_index],[0,0,1],[0,0,0,1], flags=urdfFlags, useFixedBase=False)

#LeggedRobot.printJointCart()

# Removing the collision pairs between the two gripper fingers and the bar they are attached to
if robot_index == 0 or robot_index == 2:
    p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 28, 29, 0)
    p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 26, 28, 0)
    p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 26, 29, 0)

if robot_index == 1:
    p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 27, 28, 0)
    p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 25, 27, 0)
    p.setCollisionFilterPair(LeggedRobot_bullet, LeggedRobot_bullet, 25, 28, 0)


# Initialising lists
jointIds = []
paramIds = []
joints_py = []
joints_pin = []
joints_pin_base = [0,0,0,0,0,0,0]
joints_py_current = []
EE_ID_py = [0, 0, 0, 0, 0]

# Enables a trackbar to select the max joint force
maxForceId = p.addUserDebugParameter("maxForce",0,100,100)

# This loop sets the linear and angular damping to 0 and adds all actuated joints to jointIds
for i in range(len(EE_ID_py)):
    for j in range(p.getNumJoints(LeggedRobot_bullet)):
        p.changeDynamics(LeggedRobot_bullet, j, linearDamping=0, angularDamping=0)
        info = p.getJointInfo(LeggedRobot_bullet, j)
        jointName = info[1]
        jointName = jointName.decode('UTF-8')
        jointType = info[2]
        #print(j)
        #print(jointName)
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            jointIds.append(j)
        if jointName == EE_frame_names[i]:
            EE_ID_py[i] = j

# set intitail simulation configuration
#joints_py = LeggedRobot.current_joint_config[7:]
joints_py = np.hstack((LeggedRobot.FR_leg, LeggedRobot.FL_leg, LeggedRobot.RR_leg, LeggedRobot.RL_leg, LeggedRobot.grip))
maxForce = p.readUserDebugParameter(maxForceId)

for i in range(len(joints_py)):
    p.setJointMotorControl2(LeggedRobot_bullet, jointIds[i], p.POSITION_CONTROL, joints_py[i], force=maxForce)

t = time.time()

while (time.time()- t) < 2:
        p.stepSimulation()
        time.sleep(1./500)

print("Y", abs(LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[0]].translation[2]))

p.resetBasePositionAndOrientation(LeggedRobot_bullet, [0, 0, (abs(LeggedRobot.robot_data.oMf[LeggedRobot.trunk_frame_index].translation[2]))+0.05], [0, 0, 0, 1])
print("T", abs(LeggedRobot.robot_data.oMf[LeggedRobot.trunk_frame_index].translation[2]))
print("FR pin", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[0]].translation)
print("FR py", np.array([p.getLinkState(LeggedRobot_bullet, EE_ID_py[0])[4]]))
print("GRIP pin", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[4]].translation)
print("GRIP py", np.array([p.getLinkState(LeggedRobot_bullet, EE_ID_py[4])[4]]))

# allow simulation to settle
t = time.time()
while (time.time()- t) < 2:
    time.sleep(1./500)

# Simulation camera settings
p.getCameraImage(1000,1000)
#p.resetDebugVisualizerCamera(1.0,1.25,-19.8,[0.07,0.1,0.07])
if robot_index == 0 or robot_index == 1:
    p.resetDebugVisualizerCamera(1.20,63.65,-31.4,[0.04,0.03,0.13])
if robot_index == 2:
    p.resetDebugVisualizerCamera(1.20,63.65,-31.4,[0.04,0.03,0.13])
p.setRealTimeSimulation(1)

# select the tasks that are active
LeggedRobot.setTasks(EE=True, Trunk=True, Joint="HYBRID")

# select the constraints that are active
LeggedRobot.setConstraints(foot=True, CoM=True)

# setting objectives#
"""
EE_pos_FR = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[0]].translation]).T #np.array([[0.174, -0.142, -0.32]]).T
EE_pos_FL = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[1]].translation]).T #np.array([[0.163, 0.144, -0.32]]).T
EE_pos_RR = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[2]].translation]).T #np.array([[-0.196, -0.148, -0.32]]).T
EE_pos_RL = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[3]].translation]).T #np.array([[-0.203, 0.148, -0.32]]).T
EE_pos_GRIP = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[4]].translation]).T #np.array([[0.202, 0., 0.227]]).T #base: 0.252 0. 0.207 reach 0.5 0 -0.15
"""
"""
EE_pos_FR = np.array([p.getLinkState(LeggedRobot_bullet, EE_ID_py[0])[4]]).T
EE_pos_FL = np.array([p.getLinkState(LeggedRobot_bullet, EE_ID_py[1])[4]]).T
EE_pos_RR = np.array([p.getLinkState(LeggedRobot_bullet, EE_ID_py[2])[4]]).T
EE_pos_RL = np.array([p.getLinkState(LeggedRobot_bullet, EE_ID_py[3])[4]]).T
"""
EE_pos_FR = LeggedRobot.prev_EE_pos[0].reshape(3,1)
EE_pos_FL = LeggedRobot.prev_EE_pos[1].reshape(3,1)
EE_pos_RR = LeggedRobot.prev_EE_pos[2].reshape(3,1)
EE_pos_RL = LeggedRobot.prev_EE_pos[3].reshape(3,1)
EE_pos_GRIP = LeggedRobot.prev_EE_pos[4].reshape(3,1)
#EE_pos_GRIP = np.array([p.getLinkState(LeggedRobot_bullet, EE_ID_py[4])[4]]).T


Trunk_target_pos = np.array([LeggedRobot.robot_data.oMf[LeggedRobot.trunk_frame_index].translation]).T # 0.013 0.002 0.001
EE_target_pos = [EE_pos_FR, EE_pos_FL, EE_pos_RR, EE_pos_RL, EE_pos_GRIP]


# Main while loop
while (1):

    # allow simulation to settle
    t = time.time()
    while (time.time()- t) < 2:
        time.sleep(1./500)

    base_offset = np.array([p.getLinkState(LeggedRobot_bullet, 1)[4]])

    # fetch the IMU data for potision and orientation in the world frame
    imu_state = p.getLinkState(LeggedRobot_bullet, 1)
    base_config = np.concatenate((LeggedRobot.current_joint_config[:3], np.array(imu_state[5])), axis=0)
    print(base_config)

    # get CoM position
    base_pos = imu_state[0]
    previouse_com_pos = np.add(LeggedRobot.robot_data.com[0], base_pos).tolist()
    
    # set weight mode
    if sim_id == 0:
        LeggedRobot.staticReachMode()
    """
    if sim_id == 1:
        LeggedRobot.chickenheadMode()
    """
    # finding current states
    current_gripper_pos = LeggedRobot.prev_EE_pos[4].reshape(3,) #np.array([p.getLinkState(LeggedRobot_bullet, EE_ID_py[4])[4]]).reshape(3,)
    current_trunk_pos = np.array([p.getLinkState(LeggedRobot_bullet, 1)[4]]).reshape(3,)
    
    # defining trajectories
    a1_wx200_trunk_milestones = [current_trunk_pos.tolist(), [0, 0, 0.15], [0, 0, 0.48], current_trunk_pos.tolist(), [0, -0.15, current_trunk_pos.tolist()[2]], [0, 0.15, current_trunk_pos.tolist()[2]], current_trunk_pos.tolist(), [-0.1, 0, current_trunk_pos.tolist()[2]], [0.08, 0, current_trunk_pos.tolist()[2]], current_trunk_pos.tolist()]
    #a1_wx200_milestones = [current_gripper_pos.tolist(), [0.4, 0., 0.58], [0.4, -0.25, 0.58], [0.3, -0.35, 0.58], [0, -0.35, 0.58], [-0.1, -0.46, 0.48], [-0.1, -0.35, 0.43], [0, -0.35, 0.29], [0.3, -0.35, 0.29], [0.4, -0.2, 0.29], [0.4, 0.2, 0.29], [0.3, 0.40, 0.29], [0, 0.35, 0.29], [-0.1, 0.35, 0.43], [-0.1, 0.35, 0.48], [0, 0.35, 0.58], [0.3, 0.35, 0.58], [0.4, 0.2, 0.58], [0.4, 0, 0.58]]
    a1_wx200_milestones = [current_gripper_pos.tolist(), [0.402, 0., 0.724], [0.402, 0.25, 0.724], [0.3, 0.35, 0.724], [0, 0.35, 0.724], [-0.1, 0.35, 0.48], [-0.1, 0.35, 0.43], [0, 0.37, 0.24], [0.3, 0.37, 0.24], [0.402, 0.2, 0.24], [0.402, -0.2, 0.24], [0.3, -0.37, 0.24], [0, -0.37, 0.24], [-0.1, -0.35, 0.43], [-0.1, -0.35, 0.48], [0, -0.35, 0.725], [0.3, -0.35, 0.725], [0.402, -0.2, 0.725], [0.402, 0, 0.725]]
    a1_px100_milestones = [current_gripper_pos.tolist(), [0.28, 0., 0.55], [0.28, -0.1, 0.55], [0.22, -0.2, 0.55], [0.07, -0.2, 0.55], [0., -0.2, 0.52], [0., -0.2, 0.47], [0.07, -0.2, 0.44], [0.22, -0.20, 0.44],[0.32, -0.05, 0.44], [0.32, 0.1, 0.44], [0.22, 0.2, 0.44], [0.07, 0.2, 0.44], [0, 0.2, 0.47], [0, 0.2, 0.52], [0.07, 0.2, 0.55], [0.22, 0.2, 0.55], [0.28, 0.1, 0.55], [0.28, 0, 0.55]]
    laikago_vx300_milestones = [current_gripper_pos.tolist(), [0.4, 0., 0.80], [0.4, 0.2, 0.80], [0.25, 0.38, 0.80], [-0.1, 0.38, 0.80], [-0.3, 0.38, 0.75], [-0.3, 0.38, 0.55], [-0.1, 0.38, 0.5], [0.3, 0.38, 0.5], [0.4, 0.2, 0.5], [0.4, -0.25, 0.5], [0.3, -0.38, 0.5], [-0.1, -0.38, 0.5], [-0.3, -0.38, 0.55], [-0.3, -0.38, 0.75], [-0.1, -0.38, 0.80], [0.3, -0.38, 0.80], [0.4, -0.2, 0.80], [0.4, 0, 0.80]]

    # select trajectory milestones
    if sim_id == 0:
        milestones_list = [a1_wx200_milestones, a1_px100_milestones, laikago_vx300_milestones]
    if sim_id == 1:
        milestones_list = [a1_wx200_trunk_milestones]
    milestones = milestones_list[robot_index]

    # create trajectory
    traj = trajectory.Trajectory(milestones=milestones)
    traj2 = trajectory.HermiteTrajectory()
    traj2.makeSpline(traj)
    if sim_id == 0:
        interval = 0.002
    if sim_id == 1:
        interval = 0.001
    traj_interval = np.arange(0,len(milestones),interval).tolist()
    
    # plot desired trajectory
    previouse_traj_point = np.add(traj2.eval(0), 0).tolist()
    for i in np.arange(0,len(milestones),0.1).tolist():
        current_traj_point = np.add(traj2.eval(i), 0).tolist()
        p.addUserDebugLine(previouse_traj_point, current_traj_point, lineColorRGB=[0,0,0], lineWidth=3, lifeTime=0)
        previouse_traj_point = current_traj_point

    print("CoM:")
    print(LeggedRobot.robot_data.com[0])
    print("Trunk:")
    print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.trunk_frame_index].translation)
    print("bullet:", p.getLinkState(LeggedRobot_bullet, 1)[4])
    print("\n")
    print("FR:")
    print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[0]].translation)
    print("bullet:", p.getLinkState(LeggedRobot_bullet, EE_ID_py[0])[4])
    print("\n")
    print("FL:")
    print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[1]].translation)
    print("bullet:", p.getLinkState(LeggedRobot_bullet, EE_ID_py[1])[4])
    print("\n")
    print("RR:")
    print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[2]].translation)
    print("bullet:", p.getLinkState(LeggedRobot_bullet, EE_ID_py[2])[4])
    print("\n")
    print("RL:")
    print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[3]].translation)
    print("bullet:", p.getLinkState(LeggedRobot_bullet, EE_ID_py[3])[4])
    print("\n")
    print("Grip:")
    print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[4]].translation)
    print("bullet:", p.getLinkState(LeggedRobot_bullet, EE_ID_py[4])[4])
    print(joints_py)
    
    t = time.time()
    while (time.time()- t) < 2:
            p.stepSimulation()
            time.sleep(1./500)

    LeggedRobot.initialiseWBC(np.array(p.getLinkState(LeggedRobot_bullet, 1)[5]))
    
    """
    print(LeggedRobot.robot_data.oMf[LeggedRobot.trunk_frame_index].translation)
    XY_offset = np.array([p.getLinkState(LeggedRobot_bullet, 1)[4][:2]]).reshape(2,)
    base_pos = LeggedRobot.current_joint_config[:7]
    base_pos[0] = base_pos[0] + XY_offset[0]
    base_pos[1] = base_pos[1] + XY_offset[1]
    print("base_pos",base_pos)
    print(XY_offset)
    print("current", LeggedRobot.current_joint_config)
    LeggedRobot.updateState(LeggedRobot.current_joint_config[7:], base_pos, feedback=True, running=True)
    print(LeggedRobot.robot_data.oMf[LeggedRobot.trunk_frame_index].translation)
    """
    print("start")

    start_time = time.time()
    
    for i in traj_interval:

        #print(planner_pos[i])
        
        maxForce = p.readUserDebugParameter(maxForceId)

        # Find new position of frame to move
        if sim_id == 0:
            EE_pos_GRIP = np.array([traj2.eval(i)]).T
            EE_target_pos[4] = EE_pos_GRIP
        if sim_id == 1:
            Trunk_target_pos = np.array([traj2.eval(i)]).T

        # fetch the IMU data for potision and orientation in the world frame
        imu_data = np.array(p.getLinkState(LeggedRobot_bullet, 1)[5])
        
        imu_pos = np.array(p.getLinkState(LeggedRobot_bullet, 1)[4])

        #imu_state = np.array([0,0,0,1])
        #imu = np.concatenate((imu_pos, imu_state), axis=0)
        #print("py:", p.getEulerFromQuaternion(np.array(p.getLinkState(LeggedRobot_bullet, 1)[5])))
        
        
        #base_config = np.concatenate((LeggedRobot.current_joint_config[:3], imu_state), axis=0)

        #st = time.time()
        # run the WBC to solve for the new joint configurations
        FL_leg, FR_leg, RL_leg, RR_leg, grip = LeggedRobot.runWBC(imu_data, target_cartesian_pos_EE=EE_target_pos, target_cartesian_pos_trunk=Trunk_target_pos)
        #print("pin:", LeggedRobot.Rot2Euler(LeggedRobot.robot_data.oMf[LeggedRobot.trunk_frame_index].rotation).reshape(3,))
        #print("py:", imu_pos)
        base_offset = np.array([p.getLinkState(LeggedRobot_bullet, 0)[4]])
        #print("py offset:", base_offset)
        #print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.trunk_frame_index].translation.T)

        joint_config = np.hstack((FR_leg, FL_leg, RR_leg, RL_leg, grip))

        #print(time.time()-st)

        # update simulation with new joint configurations
        for ii in range(len(joints_py)):
            p.setJointMotorControl2(LeggedRobot_bullet, jointIds[ii], p.POSITION_CONTROL, joint_config[ii], force=maxForce)

        current_time = time.time() - start_time

        # visually track the CoM
        base_pos = imu_state[0]
        current_com_pos = np.add(LeggedRobot.robot_data.com[0], base_pos).tolist()

        #p.addUserDebugLine(previouse_com_pos, current_com_pos, lineColorRGB=[255,0,0], lineWidth=10, lifeTime=0, parentObjectUniqueId=100)

        previouse_com_pos = current_com_pos

        """ Data Collection """
        if data_collection == True:
            # collect data in dict
            data_dict["Time"].append(current_time)
            data_dict["target x"].append(float(EE_target_pos[4][0]))
            data_dict["target y"].append(float(EE_target_pos[4][1]))
            data_dict["target z"].append(float(EE_target_pos[4][2]))
            data_dict["real x"].append(p.getLinkState(LeggedRobot_bullet, EE_ID_py[4])[4][0])
            data_dict["real y"].append(p.getLinkState(LeggedRobot_bullet, EE_ID_py[4])[4][1])
            data_dict["real z"].append(p.getLinkState(LeggedRobot_bullet, EE_ID_py[4])[4][2])
            

    while (1):
        print("done")
        print("CoM:")
        print(LeggedRobot.robot_data.com[0])
        print("Trunk:")
        print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.trunk_frame_index].translation)
        print("bullet:", p.getLinkState(LeggedRobot_bullet, 1)[4])
        print("\n")
        print("FR:")
        print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[0]].translation)
        print("bullet:", p.getLinkState(LeggedRobot_bullet, EE_ID_py[0])[4])
        print("\n")
        print("FL:")
        print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[1]].translation)
        print("bullet:", p.getLinkState(LeggedRobot_bullet, EE_ID_py[1])[4])
        print("\n")
        print("RR:")
        print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[2]].translation)
        print("bullet:", p.getLinkState(LeggedRobot_bullet, EE_ID_py[2])[4])
        print("\n")
        print("RL:")
        print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[3]].translation)
        print("bullet:", p.getLinkState(LeggedRobot_bullet, EE_ID_py[3])[4])
        print("\n")
        print("Grip:")
        print("pin:", LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[4]].translation)
        print("bullet:", p.getLinkState(LeggedRobot_bullet, EE_ID_py[4])[4])
        print(joints_py)

        p.stepSimulation()
        break
    break

# write data to CSV
if data_collection == True:
    data = pd.DataFrame(data_dict)
    data.to_csv(data_name,index=False)
    print("data saved")

# Cleanup
p.disconnect()
