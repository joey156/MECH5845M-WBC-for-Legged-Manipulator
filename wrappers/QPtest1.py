import numpy as np
import qpsolvers as qp
from Robot_Wrapper import RobotModel

urdf = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_wx200.urdf"

LeggedRobot = RobotModel(urdf)

LeggedRobot.neutralConfig()

#targets
com_target_pos = np.array([[1, 1, 1]]).T

planner_pos, planner_vel = LeggedRobot.posAndVelTargetsCoM(com_target_pos)

lower_vel_lim, upper_vel_lim = LeggedRobot.jointVelLimitsArray()
lower_pos_lim, upper_pos_lim = LeggedRobot.jointPosLimitsArray()

lb = np.concatenate((lower_vel_lim.T, lower_pos_lim), axis=0)
ub = np.concatenate((upper_vel_lim.T, upper_pos_lim), axis=0)
