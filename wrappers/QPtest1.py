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


lower_vel_lim, upper_vel_lim = LeggedRobot.jointVelLimitsArray()
lower_pos_lim, upper_pos_lim = LeggedRobot.jointPosLimitsArray()

lb = np.concatenate((lower_vel_lim.T, lower_pos_lim), axis=0).reshape((52,))
#lb = lower_vel_lim.reshape(26,)
#print(lb.shape)

ub = np.concatenate((upper_vel_lim.T, upper_pos_lim), axis=0).reshape((52,))
#ub = upper_vel_lim.reshape(26,)
#print(ub)

print(lb)
A = np.dot(LeggedRobot.comJ.T, LeggedRobot.comJ)
#print(LeggedRobot.comJ)
#print(A)
b = LeggedRobot.qpCartesianB(planner_pos[499], planner_vel[499]).reshape((3,))
#print(b)
b = np.dot(b.T, LeggedRobot.comJ)
#print(b)
print(b.shape)

qp = QProblemB(26)

options = Options()
options.enableFlippingBounds = BooleanType.FALSE
#options.initialStatusBounds = SubjectToStatus.INACTIVE
options.numRefinementSteps = 10

qp.setOptions(options)

nWSR = np.array([100000])
qp.init(A, b, lb, ub, nWSR)

#print("\nnWSR = %d\n\n"%nWSR)

xOpt = np.zeros((26,))
qp.getPrimalSolution(xOpt)
print("\nxOpt = [ %e, %e ]; objCal = %e\n\n" %(xOpt[0], xOpt[1], qp.getObjVal()))
print(xOpt)
