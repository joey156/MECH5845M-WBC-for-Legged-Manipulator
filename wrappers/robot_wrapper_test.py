import sys
sys.path.append("/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/wrappers/")
from Robot_Wrapper import RobotModel
#from scipy.spatial.transform import Rotation as R
import numpy as np


urdf = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_wx200.urdf"

LeggedRobot = RobotModel(urdf)

LeggedRobot.neutralConfig()


r = LeggedRobot.EndEffectorJacobians()
lower_pos_lim, upper_pos_lim = LeggedRobot.jointPosLimitsArray()
lower_vel_lim, upper_vel_lim = LeggedRobot.jointVelLimitsArray()
print(lower_pos_lim)
print("\n", lower_pos_lim.shape)
x = LeggedRobot.robot_data.oMi[19].rotation
x = np.array(x)
print(x)

x = LeggedRobot.Rot2Euler(x)
com_target_pos = np.array([[1, 1, 1]]).T
com_target_vel = np.array([[1, 1, 1]]).T
EE_pos = np.array([[1,1,1]]).T
EE_vel = np.array([[1,1,1,1,1,1]]).T
print(EE_pos)
EE_target_pos = [EE_pos, EE_pos, EE_pos, EE_pos, EE_pos]
EE_target_vel = [EE_vel, EE_vel, EE_vel, EE_vel, EE_vel]
LeggedRobot.cartesianTargetsEE(EE_target_pos, EE_target_vel)
print(LeggedRobot.cartesian_targetsEE)
LeggedRobot.cartesianTargetCoM(com_target_pos, com_target_vel)
print(LeggedRobot.cartesian_targetsCoM)
print("success")
print(LeggedRobot.robot_data.oMi[19].translation.shape)
planner_pos, planner_vel = LeggedRobot.posAndVelTargetsCoM(com_target_pos)
print(planner_vel[0])
