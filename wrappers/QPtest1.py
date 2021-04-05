import numpy as np
import qpsolvers as qp
from Robot_Wrapper import RobotModel
from qpoases import PyQProblemB as QProblemB
from qpoases import PyBooleanType as BooleanType
from qpoases import PySubjectToStatus as SubjectToStatus
from qpoases import PyOptions as Options

urdf = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_wx200.urdf"

LeggedRobot = RobotModel(urdf)

LeggedRobot.neutralConfig()

#targets
com_target_pos = np.array([[2, 2, 2]]).T

planner_pos, planner_vel = LeggedRobot.posAndVelTargetsCoM(com_target_pos)

EE_pos_FL = np.array([[0.163, 0.144, -0.326]]).T
EE_pos_FR = np.array([[0.174, -0.142, -0.329]]).T
EE_pos_RL = np.array([[-0.203, 0.148, -0.319]]).T
EE_pos_RR = np.array([[-0.196, -0.148, -0.32]]).T
EE_pos_GRIP = np.array([[0.453, 0., 0.363]]).T
EE_vel = np.array([[0,0,0,0,0,0]]).T
EE_target_pos = [EE_pos_FL, EE_pos_FR, EE_pos_RL, EE_pos_RR, EE_pos_GRIP]
EE_target_vel = [EE_vel, EE_vel, EE_vel, EE_vel, EE_vel]

#lims
lower_vel_lim, upper_vel_lim = LeggedRobot.jointVelLimitsArray()
lower_pos_lim, upper_pos_lim = LeggedRobot.jointPosLimitsArray()

lb = np.concatenate((lower_vel_lim.T, lower_pos_lim), axis=0).reshape((52,))
#lb = lower_vel_lim.reshape(26,)
#print(lb.shape)

ub = np.concatenate((upper_vel_lim.T, upper_pos_lim), axis=0).reshape((52,))
#ub = upper_vel_lim.reshape(26,)
#print(ub)

#print(lb)


A = LeggedRobot.qpCartesianA()
H = np.dot(A.T, A)
#print(LeggedRobot.comJ)
print(H)
b = LeggedRobot.qpCartesianB(planner_pos[499], planner_vel[499], EE_target_pos, EE_target_vel).reshape((33,))
#print(b)
g = np.dot(b.T, A)
#print(b)
print(b.shape)

qp = QProblemB(26)

options = Options()
options.enableFlippingBounds = BooleanType.FALSE
#options.initialStatusBounds = SubjectToStatus.INACTIVE
options.numRefinementSteps = 10

qp.setOptions(options)

nWSR = np.array([100000])
qp.init(H, g, lb, ub, nWSR)

#print("\nnWSR = %d\n\n"%nWSR)

xOpt = np.zeros((26,))
qp.getPrimalSolution(xOpt)
print("\nxOpt = [ %e, %e ]; objCal = %e\n\n" %(xOpt[0], xOpt[1], qp.getObjVal()))
print(xOpt)
