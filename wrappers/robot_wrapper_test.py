import sys
sys.path.append("/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/wrappers/")
from Robot_Wrapper import RobotModel
#from scipy.spatial.transform import Rotation as R
import numpy as np
import pinocchio as pin
import math


large_width = 400
np.set_printoptions(linewidth=large_width)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=1200)

urdf_path = "/home/joey156/Documents/Robot_Descriptions/urdf/a1_wx200.urdf"

# initialise the RobotModel class
EE_frame_names = ["FR_foot_fixed", "FL_foot_fixed", "RR_foot_fixed", "RL_foot_fixed", "gripper_bar"]
EE_joint_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint", "gripper"]
hip_joint_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
#LeggedRobot = RobotModel(urdf_path, "FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed", "gripper_bar", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint", "gripper", "waist", "imu_joint")
LeggedRobot = RobotModel(urdf_path, EE_frame_names, EE_joint_names, "waist", "imu_joint", "FR_hip_joint", hip_joint_names)#LeggedRobot.neutralConfig()


LeggedRobot.EndEffectorJacobians()
#print(LeggedRobot.end_effector_jacobians)
lower_pos_lim, upper_pos_lim = LeggedRobot.jointPosLimitsArray()
lower_vel_lim, upper_vel_lim = LeggedRobot.jointVelLimitsArray()
print(lower_vel_lim)
#print("\n", lower_pos_lim.shape)
x = LeggedRobot.robot_data.oMi[19].rotation
x = np.array(x)
#print(x)
#print(LeggedRobot.current_joint_config)
x = LeggedRobot.Rot2Euler(x)


com_target_pos = np.array([[1, 1, 1]]).T
planner_pos, planner_vel = LeggedRobot.posAndVelTargetsCoM(com_target_pos)
EE_pos_FL = np.array([[0.263, 0.144, -0.326]]).T
EE_pos_FR = np.array([[0.174, -0.142, -0.329]]).T
EE_pos_RL = np.array([[-0.203, 0.148, -0.319]]).T
EE_pos_RR = np.array([[-0.196, -0.148, -0.32]]).T
EE_pos_GRIP = np.array([[0.453, 0., 0.363]]).T
EE_vel = np.array([[0,0,0,0,0,0]]).T
#print(EE_pos)
EE_target_pos = [EE_pos_FL, EE_pos_FR, EE_pos_RL, EE_pos_RR, EE_pos_GRIP]
EE_target_vel = [EE_vel, EE_vel, EE_vel, EE_vel, EE_vel]
LeggedRobot.cartesianTargetsEE(EE_target_pos, EE_target_vel)
#print(LeggedRobot.cartesian_targetsEE)

#print(LeggedRobot.qpCartesianB(planner_pos[1], planner_vel[1], EE_target_pos, EE_target_vel))


#Ji = pin.getFrameJacobian(LeggedRobot.robot_model, LeggedRobot.robot_data, 10, pin.LOCAL_WORLD_ALIGNED)

#idet = np.linalg.det(np.dot(Ji, Ji.T))
#isqrt = math.sqrt(idet)
#diffr = np.ones((3,)) * isqrt
#print(np.dot(Ji, Ji.T))
#print(idet)
#print(isqrt)
#print(np.diff(diffr))

#pin.pinocchio_pywrap.KinematicLevel


#print(LeggedRobot.cartesian_targetsCoM)
#print("success")
#print("this")
#print(LeggedRobot.robot_data.oMi[].translation)
#print(LeggedRobot.robot_data.oMf[19].translation)
#print(LeggedRobot.robot_data.oMf[27].translation)
#print(LeggedRobot.robot_data.oMf[35].translation)
#print(LeggedRobot.robot_data.oMf[53].translation)
#planner_pos, planner_vel = LeggedRobot.posAndVelTargetsCoM(com_target_pos)
#print(planner_vel[0])
#.printJointCart()
q = np.array([ 0., 0., 0., -0.001, 0.012, -0.001, 1., 0.007, 0.868, -1.168, -0.001, 0.869, -1.169, 0.007, 0.823, -1.094, -0.001, 0.823, -1.094, -0.002, -2.133, 0.945, 1.112, 0.002, 0., 0., 0.])

LeggedRobot.updateState(q, feedback=False)
print(LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[0]].translation)

q = np.array([ 0., 0., 0., -0.001, 0.012, -0.001, 1., 0.007, 0.868, -1.168, -0.001, 1, -1.169, 0.007, 0.823, -1.094, -0.001, 0.823, -1.094, -0.002, -2.133, 0.945, 1.112, 0.002, 0., 0., 0.])

LeggedRobot.updateState(q, feedback=False)
print(LeggedRobot.robot_data.oMf[LeggedRobot.end_effector_index_list_frame[0]].translation)


#LeggedRobot.printJointCart()

#print(pin.getJointJacobian(LeggedRobot.robot_model, LeggedRobot.robot_data, 1, pin.LOCAL_WORLD_ALIGNED))

#print(LeggedRobot.robot_model.getFrameId("imu_joint", pin.FIXED_JOINT))
#print(pin.getFrameJacobian(LeggedRobot.robot_model, LeggedRobot.robot_data, 63, pin.LOCAL_WORLD_ALIGNED))
#print(LeggedRobot.robot_model.getFrameId("floating_base", pin.JOINT))
#print(LeggedRobot.robot_model.getFrameId("RR_foot_fixed", pin.FIXED_JOINT))
#print(LeggedRobot.robot_model.getFrameId("ee_gripper", pin.FIXED_JOINT))
#print(LeggedRobot.robot_model.getFrameId("gripper_bar", pin.FIXED_JOINT))
#print(LeggedRobot.qpCartesianA())
#print(LeggedRobot.comJ)

#A = LeggedRobot.qpCartesianA()
#print(A)
#print(LeggedRobot.robot_model.getJointId("waist"))
#print(len(LeggedRobot.robot_model.subtrees[14]))
#print(LeggedRobot.robot_model.nv)
