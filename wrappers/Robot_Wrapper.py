import pinocchio as pin
import numpy as np
#from scipy.spatial.transform import Rotation as R
import math

#large_width = 400
#np.set_printoptions(linewidth=large_width)
#np.set_printoptions(precision=3)
#np.set_printoptions(suppress=True)

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
        self.comJacobian()
        self.cartesian_targetsEE = 0
        self.cartesian_targetsCoM = 0
        self.end_effector_jacobians = 0
        
        self.sampling_time = 0.001 #in seconds (1ms)
        self.end_effector_index_list_v = [8, 11, 14, 17, 23]
        self.end_effecotr_index_list_oMi = [4, 7, 10, 13, 19]

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

    def EndEffectorJacobians(self): # This works with the current model configuration
        E_J = np.transpose(pin.getJointJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_v[0], pin.WORLD))
        for i in range(len(self.end_effector_index_list_v)-1):
            J = np.transpose(pin.getJointJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_v[i+1], pin.WORLD))
            E_J = np.concatenate((E_J, J), axis = 1)
        E_J = np.transpose(E_J)
        W = np.identity(30) # Later this can be used to weight each of the cartisian tasks
        E_J = np.dot(W, E_J)
        return E_J
        #print(self.end_effector_jacobians)
            
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

    def comJacobian(self):
        self.comJ = pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.current_joint_config)

    def qpCartisianA(self):
        A = np.concatenate((self.end_effector_jacobians, self.comJ), axis=0)
        return A

    def cartesianTargetsEE(self, target_cartesian_pos, target_cartesian_vel):
        K_cart = np.identity(26)
        target_list = []
        if np.sum(target_cartesian_pos) == 0 and np.sum(target_cartesian_vel) == 0:
            self.cartesian_targetsEE = np.zeros((30,1))
        else:
            for i in range(len(self.end_effector_index_list)):
                if target_cartesian_pos[i] == 0 and target_cartesian_vel[i] == 0:
                    target_list[i] = np.zeros((6,1))
                else:
                    rot = LeggedRobot.robot_data.oMi[self.end_effector_inde_list_oMi[i]].rotation
                    rot = self.Rot2Euler(rot)
                    #rot = np.array(rot)
                    #rot = R.from_matrix(rot)
                    #rot = np.array([x.as_euler("xyz")]).T
                    x = target_cartesian_pos[i] - self.robot_data.oMi[self.end_effector_index_list_oMi[i]].translation
                    target_list[i] = np.concatenate(((target_cartesian_vel[i] + np.dot(K_cart, x)), rot), axis=0)
            self.cartesian_targetsEE = target_list[0]
            for i in range(len(target_list)-1):
                self.cartesian_targetsEE = np.concatenate((self.cartesian_targetsEE,target_list[i+1]), axis=0)

    def cartesianTargetCoM(self, target_cartesian_pos, target_cartesian_vel):
        K_cart = np.identity(3)
        if np.sum(target_cartesian_pos) == 0 and np.sum(target_cartesian_vel) == 0:
            self.cartesian_targetsCoM = np.zeros((3,1))
        else:
            self.cartesian_targetsCoM = target_cartesian_vel + np.dot(K_cart, (target_cartesian_pos-self.robot_data.com[0]))


    #Debugging functions
    def printJointCart(self):
        print("Cartisian Joint Placements in the World Frame:")
        for name, oMi in zip(self.joint_names, self.oMi):
            print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)))
        print("\n")

    def neutralConfig(self):
        q = pin.neutral(self.robot_model)
        self.updateState(q)
        #self.printJointCart()

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
        


    #Helper functions
    def Rot2Euler(self, Rot):
        roll = math.atan2(Rot[2,1],Rot[2,2])
        pitch = math.atan2(-Rot[2,0],math.sqrt(math.pow(Rot[2,1],2)+math.pow(Rot[2,2],2)))
        yaw = math.atan2(Rot[1,0],Rot[0,0])
        return np.array([[roll,pitch,yaw]]).T


#urdf = "/home/joey156/Disso_ws/MECH5845M-WBC-for-Legged-Manipulator/Robot_Descriptions/urdf/a1_wx200.urdf"

#LeggedRobot = RobotModel(urdf)

#LeggedRobot.neutralConfig()

#LeggedRobot.EndEffectorJacobians()

#LeggedRobot.printJointCart()
