import sys
sys.path.append("../wrappers/")
from Robot_Wrapper import RobotModel


urdf = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_wx200.urdf"

LeggedRobot = RobotModel(urdf)

LeggedRobot.neutralConfig()

LeggedRobot.printJ()

LeggedRobot.printJ(0)
LeggedRobot.printJ(1)
LeggedRobot.printJ(2)
LeggedRobot.printJ(3)
LeggedRobot.printJ(4)
LeggedRobot.printJ(5)
LeggedRobot.printJ(6)
LeggedRobot.printJ(7)
LeggedRobot.printJ(8)
LeggedRobot.printJ(9)
LeggedRobot.printJ(10)
LeggedRobot.printJ(11)
LeggedRobot.printJ(12)
LeggedRobot.printJ(13)
LeggedRobot.printJ(14)
LeggedRobot.printJ(15)
LeggedRobot.printJ(16)
LeggedRobot.printJ(17)
LeggedRobot.printJ(18)
LeggedRobot.printJ(19)
LeggedRobot.printJ(20)
LeggedRobot.printJ(21)
LeggedRobot.printJ(22)
LeggedRobot.printJ(23)
LeggedRobot.printJ(24)
LeggedRobot.printJ(25)

LeggedRobot.printJointJ(4)
#LeggedRobot.printCoMJ()

LeggedRobot.EndEffectorJacobians([4, 7, 10, 13, 19])

print(LeggedRobot.robot_model.velocityLimit)

