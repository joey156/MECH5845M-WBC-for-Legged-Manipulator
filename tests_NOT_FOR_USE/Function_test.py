import pinocchio as pin
import numpy as np
import math

large_width = 400
np.set_printoptions(linewidth=large_width)
np.set_printoptions(suppress = True)

#global variables
end_effector_index = [8, 11, 14, 17, 23]
x = [1, 2, 3, 4]

# model setup
urdf = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_wx200.urdf"
robot_model = pin.buildModelFromUrdf(urdf)
robot_data = robot_model.createData()

# setting the model to a neutral stance
current_joint_config = pin.neutral(robot_model)
pin.forwardKinematics(robot_model, robot_data, current_joint_config)
J = pin.computeJointJacobians(robot_model, robot_data)

# finding the end effector jacobians
end_effector_jacobians = pin.getJointJacobian(robot_model, robot_data, end_effector_index[0], pin.WORLD)
end_effector_jacobians = np.transpose(end_effector_jacobians)
for i in range((len(end_effector_index) -1)):
    y = 1
    J = pin.getJointJacobian(robot_model, robot_data, end_effector_index[y], pin.WORLD)
    J = np.transpose(J)
    end_effector_jacobians = np.concatenate((end_effector_jacobians, J), axis=1)
    y = y+1
#end_effector_jacobians = np.transpose(end_effector_jacobians)
#W = np.identity(30)
#end_effector_jacobians = np.dot(W, end_effector_jacobians)

print("success")
