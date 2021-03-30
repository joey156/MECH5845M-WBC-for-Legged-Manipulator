import pinocchio as pin
import numpy as np

large_width = 400
np.set_printoptions(linewidth=large_width)
np.set_printoptions(precision=3)

class RobotModel:
    def __init__(self, urdf_path):
        #initialise pinocchio model and data
        self.robot_model = pin.buildModelFromUrdf(urdf_path)
        self.robot_data = self.robot_model.createData()
        self.joint_names = self.robot_model.names
        self.no_DoF = self.robot_model.nv
        self.no_config = self.robot_model.nq
        #set robot to a neutral stance and initalise parameters
        self.current_joint_config = pin.neutral(self.robot_model)
        self.neutralConfig()
        
        self.sampling_time = 0.001 #in seconds (1ms)

    def updateState(self, joint_config):
        #update robot configuration
        pin.forwardKinematics(self.robot_model, self.robot_data, joint_config)
        #update current joint configurations, joint jacobians and absolute joint placements in the world frame
        self.previouse_joint_config = self.current_joint_config
        self.current_joint_config = joint_config
        self.J = pin.computeJointJacobians(self.robot_model, self.robot_data)
        self.oMi = self.robot_data.oMi
        
    def jointVelocitiestoConfig(self, joint_vel, updateModel=False):
        new_config = pin.integrate(self.robot_model, self.current_joint_config, joint_vel)
        if updateModel == True:
            self.updateStates(new_config)
        if updateModel == False:
            return new_config

    def EndEffectorJacobians(self, end_effector_index_list): # This works with the current model configuration
        self.end_effector_jacobians = np.transpose(pin.getJointJacobian(self.robot_model, self.robot_data, end_effector_index_list[0], pin.WORLD))
        for i in range(len(end_effector_index_list)-1):
            J = np.transpose(pin.getJointJacobian(self.robot_model, self.robot_data, end_effector_index_list[i+1], pin.WORLD))
            self.end_effector_jacobians = np.concatenate((self.end_effector_jacobians, J), axis = 1)
        self.end_effector_jacobians = np.transpose(self.end_effector_jacobians)
        W = np.identity(30)
        print(self.end_effector_jacobians)
        print(W)
            
    def jointVelLimitsArray(self): # returns an array for the upper and lower joint velocity limits which will be used for QP
        vel_lim = self.robot_model.velocityLimit
        lower_vel_lim = -vel_lim[np.newaxis]
        upper_vel_lim = vel_lim[np.newaxis]
        return lower_vel_lim, upper_vel_lim

    def jointPosLimitsArray(self): # returns an array for the upper and lower joint position limits, these have been turned into velocity limits
        for i in range(len(self.robot_model.lowerPositionLimit)):
            if np.isinf(self.robot_model.lowerPositionLimit[i]):
                self.robot_model.lowerPositionLimit[i] = 0
        lower_pos_lim = np.transpose(self.robot_model.lowerPositionLimit[np.newaxis])
        upper_pos_lim = self.robot_model.upperPositionLimit

        K_lim = np.identity(27)*0.5
        lower_pos_lim = np.dot(K_lim,(lower_pos_lim - np.transpose(self.current_joint_config[np.newaxis])))*(1/self.sampling_time)
        upper_pos_lim = np.dot(K_lim,(upper_pos_lim - np.transpose(self.current_joint_config[np.newaxis])))*(1/self.sampling_time)

        lower_pos_lim = np.delete(lower_pos_lim, 0, 0)
        upper_pos_lim = np.delete(upper_pos_lim, 0, 0)
        
        return lower_pos_lim, upper_pos_lim




    #Debugging functions
    def printJointCart(self):
        print("Cartisian Joint Placements in the World Frame:")
        for name, oMi in zip(self.joint_names, self.oMi):
            print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)))
        print("\n")

    def neutralConfig(self):
        q = pin.neutral(self.robot_model)
        self.updateState(q)
        self.printJointCart()

    def printJ(self, joint_index=None):
        if (joint_index != None):
            print(self.J[:, joint_index])
        else:
            print(self.J)

    def printJointJ(self, joint_index):
        print(pin.getJointJacobian(self.robot_model, self.robot_data, joint_index, pin.WORLD))
        print("\n")

    def printCoMJ(self):
        print(pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.current_joint_config))
        
    
    
