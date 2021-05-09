import pinocchio as pin
import numpy as np
import math
from QP_Wrapper import QP
from klampt.model import trajectory

large_width = 400
np.set_printoptions(linewidth=large_width)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

class RobotModel:
    def __init__(self, urdf_path, FL_frame, FR_frame, RL_frame, RR_frame, G_frame, FL_joint, FR_joint, RL_joint, RR_joint, G_joint, G_base, imu):
        #initialise pinocchio model and data
        self.robot_model = pin.buildModelFromUrdf(urdf_path)
        self.robot_data = self.robot_model.createData()
        self.joint_names = self.robot_model.names
        self.no_DoF = self.robot_model.nv
        self.no_config = self.robot_model.nq
        #set robot to a neutral stance and initalise parameters
        #self.stand_joint_config = np.array([0,0,0,0,0,0,1,0.037199,0.660252,-1.200187,-0.028954,0.618814,-1.183148,0.048225,0.690008,-1.254787,-0.050525,0.661355,-1.243304, 0, -1.6, 1.6, 0, 0, 0, 0.02, -0.02])
        #self.stand_joint_config = np.array([ 0.,-0.,0.,0.,0.,0.,1.,0.,0.625,-0.754,-0.,0.625,-0.754,0.,0.625,-0.754,-0.,0.625,-0.754,0.,0.068,0.287,-0.36,-0.,0.,0.,0.])
        self.current_joint_config = 0
        self.EE_frame_names = [FL_frame, FR_frame, RL_frame, RR_frame, G_frame]
        self.EE_joint_names = [FL_joint, FR_joint, RL_joint, RR_joint, G_joint]
        self.arm_base_id = self.robot_model.getJointId(G_base)
        self.n_velocity_dimensions = self.robot_model.nv
        self.n_of_EE = 5
        self.end_effector_index_list_frame = [] #[11, 19, 27, 35, 53]
        self.end_effector_index_list_joint = []
        for i in range(len(self.EE_joint_names)):
            ID = self.robot_model.getFrameId(self.EE_frame_names[i], pin.FIXED_JOINT)
            self.end_effector_index_list_frame.append(ID)
            ID = self.robot_model.getJointId(self.EE_joint_names[i])
            self.end_effector_index_list_joint.append(ID)

        print(self.end_effector_index_list_frame)
        print(self.end_effector_index_list_joint)

        self.sampling_time = 0.002 #in seconds (2ms)
        self.trunk_frame_index= self.robot_model.getFrameId(imu, pin.FIXED_JOINT)

        # cartesian task weights
        self.com_weight = np.identity(3) * 20#1.2
        self.trunk_weight = np.identity(6) * 15 #1
        self.FR_weight = np.identity(6) * 20 #0.8
        self.FL_weight = np.identity(6) * 20 #0.8
        self.RR_weight = np.identity(6) * 20 #0.8
        self.RL_weight = np.identity(6) * 20 #0.8
        self.grip_weight = np.identity(6) * 15 #1
        self.EE_weight = [self.FR_weight, self.FL_weight, self.RR_weight, self.RL_weight, self.grip_weight]

        # cartesian proportional gains
        self.com_gain = np.identity(3)* 1.5 #1.631
        self.trunk_gain = np.identity(6)* 1 #0.5425
        self.EE_gain = np.identity(6)* 1.2 #1.458
        
        #self.neutralConfig()
        #self.updateState(self.stand_joint_config, feedback=False)
        self.comJacobian()
        self.cartesian_targetsEE = 0
        self.cartesian_targetsCoM = 0
        self.end_effector_jacobians = 0
        self.cartesian_targetsTrunk = 0
        self.setInitialState()


    def setInitialState(self):
        
        # seta neutral configuration
        q = pin.neutral(self.robot_model)
        self.updateState(q, feedback=False)
        # find initial cartesian position of end effectors and trunk
        EE_pos_FL = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[0]].translation]).T
        EE_pos_FR = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[1]].translation]).T 
        EE_pos_RL = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[2]].translation]).T
        EE_pos_RR = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[3]].translation]).T 
        EE_pos_GRIP = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[4]].translation]).T
        Trunk_target_pos = np.array([[self.robot_data.oMf[self.trunk_frame_index].translation]]).T
        # find initial cartesisn potision of the CoM
        com_pos = np.array([self.robot_data.com[0]]).T
        # set desired velocities for setting the initial state
        Trunk_target_vel = np.array([[0,0,0,0,0,0]]).T
        EE_vel = np.array([[0,0,0,0,0,0]]).T
        com_vel = np.array([[0, 0, 0]]).T
        # setup array to hold to positions and velocities for the end effectors 
        EE_target_pos = [EE_pos_FL, EE_pos_FR, EE_pos_RL, EE_pos_RR, EE_pos_GRIP]
        EE_target_vel = [EE_vel, EE_vel, EE_vel, EE_vel, EE_vel]

        """ Setting up trajectories to desired initial configuration """
        multiplier_F = np.identity(3)
        multiplier_R = np.identity(3)
        multiplier_G = np.identity(3)
        multiplier_F[2,2] = 0.65
        multiplier_R[2,2] = 0.65
        multiplier_G[2,2] = 0.65

        EE_G_pos_2 = EE_pos_GRIP.reshape((3,)).tolist()
        print(EE_G_pos_2)
        EE_G_pos_2[0] = self.robot_data.oMi[self.arm_base_id].translation[0]
        print(EE_G_pos_2)
        
        # set trajecotry milestones
        EE_FL_milestones = [EE_pos_FL.reshape((3,)).tolist(), np.dot(EE_pos_FL.reshape((3,)), multiplier_F).tolist()]
        EE_FR_milestones = [EE_pos_FR.reshape((3,)).tolist(), np.dot(EE_pos_FR.reshape((3,)), multiplier_F).tolist()]
        EE_RL_milestones = [EE_pos_RL.reshape((3,)).tolist(), np.dot(EE_pos_RL.reshape((3,)), multiplier_R).tolist()]
        EE_RR_milestones = [EE_pos_RR.reshape((3,)).tolist(), np.dot(EE_pos_RR.reshape((3,)), multiplier_R).tolist()]
        EE_G_milestones = [EE_pos_GRIP.reshape((3,)).tolist(), np.dot(EE_G_pos_2, multiplier_G).tolist()]
        # setting up the trajectories for the feed end effectors
        EE_FL_traj = trajectory.Trajectory(milestones=EE_FL_milestones)
        EE_FR_traj = trajectory.Trajectory(milestones=EE_FR_milestones)
        EE_RL_traj = trajectory.Trajectory(milestones=EE_RL_milestones)
        EE_RR_traj = trajectory.Trajectory(milestones=EE_RR_milestones)
        EE_G_traj = trajectory.Trajectory(milestones=EE_G_milestones)
        EE_traj = [EE_FL_traj, EE_FR_traj, EE_RL_traj, EE_RR_traj, EE_G_traj]

        
        
        # set the trajectory interval
        trajectory_interval = np.arange(0,len(EE_FL_milestones), 0.001).tolist()

        """ Solving QP until desired initial configuration is reached """
        for i in trajectory_interval:

            # set new desired position for each foot
            for ii in range(len(EE_traj)):
                target_pos = EE_traj[ii]
                EE_target_pos[ii] = np.array(target_pos.eval(i)).reshape(3,1)

            # find joint limits
            lower_vel_lim, upper_vel_lim = self.jointVelLimitsArray(True)
            lower_pos_lim, upper_pos_lim = self.jointPosLimitsArray(True)
            lb = np.concatenate((lower_vel_lim.T, lower_pos_lim), axis=0).reshape(((self.n_velocity_dimensions*2),))
            ub = np.concatenate((upper_vel_lim.T, upper_pos_lim), axis=0).reshape(((self.n_velocity_dimensions*2),))

            # find A and b for QP
            A = self.qpCartesianA()
            b = self.qpCartesianB(com_pos, com_vel, EE_target_pos, EE_target_vel, Trunk_target_pos, Trunk_target_vel).reshape((39,))
            # solver QP
            qp = QP(A, b, lb, ub)
            qp_vel = qp.solveQP()

            # find the new joint angles from the QP optimised joint velocities
            joint_config = self.jointVelocitiestoConfig(qp_vel, True)

            #self.updateState(joint_config, feedback=False)
            
        print("Initial state set successfully")

        for i in range(len(self.current_joint_config)):
            if i < 6:
                self.current_joint_config[i] = 0

        joint_config = self.current_joint_config
        self.updateState(joint_config, feedback=False)
            
                
    def updateState(self, joint_config, base_config=0, feedback=True): # put floatig base info here
        if feedback == True:
            config = np.concatenate((base_config, joint_config), axis=0)

        else:
            config = joint_config
        
        #update robot configuration
        pin.forwardKinematics(self.robot_model, self.robot_data, config)
        #update current joint configurations, joint jacobians and absolute joint placements in the world frame
        self.previouse_joint_config = self.current_joint_config
        self.current_joint_config = config
        
        self.J = pin.computeJointJacobians(self.robot_model, self.robot_data)
        self.comJacobian()
        #print("\n\n")
        #print(self.robot_data.com[0])
        #print("\n\n")
        pin.framesForwardKinematics(self.robot_model, self.robot_data, config)
        self.oMi = self.robot_data.oMi
        #print(joint_config)
        
    def jointVelocitiestoConfig(self, joint_vel, updateModel=False): # add floating base stuff
        new_config = pin.integrate(self.robot_model, self.current_joint_config, joint_vel)
        if updateModel == True:
            self.updateState(new_config, feedback=False)
        if updateModel == False:
            return new_config

    def EndEffectorJacobians(self): # This works with the current model configuration
        self.end_effector_jacobians = np.dot(np.transpose(pin.getFrameJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_frame[0], pin.LOCAL_WORLD_ALIGNED)), self.EE_weight[0])
        for i in range(len(self.end_effector_index_list_frame)-1):
            if i < 3:
                J = np.transpose(pin.getFrameJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_frame[i+1], pin.LOCAL_WORLD_ALIGNED))
            else:
                J = np.transpose(pin.getFrameJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_frame[i+1], pin.LOCAL_WORLD_ALIGNED))
            J = np.dot(J, self.EE_weight[i+1])
            self.end_effector_jacobians = np.concatenate((self.end_effector_jacobians, J), axis = 1)
        self.end_effector_jacobians = np.transpose(self.end_effector_jacobians)

    def TrunkJacobian(self):
        J = pin.getFrameJacobian(self.robot_model, self.robot_data, self.trunk_frame_index, pin.WORLD)
        self.trunkJ = np.dot(self.trunk_weight, J)
        
            
    def jointVelLimitsArray(self, initial_config=False): # returns an array for the upper and lower joint velocity limits which will be used for QP
        vel_lim = self.robot_model.velocityLimit
        
        for i in range(len(vel_lim)):
            if (np.isinf(vel_lim[i])):
                vel_lim[i] = 10
            if i >= (self.end_effector_index_list_joint[4] - 2 + 6):
                vel_lim[i] = 0
 
            
        lower_vel_lim = -vel_lim[np.newaxis]
        upper_vel_lim = vel_lim[np.newaxis]
        return lower_vel_lim, upper_vel_lim

    def jointPosLimitsArray(self, initial_config=False): # returns an array for the upper and lower joint position limits, these have been turned into velocity limits
        for i in range(len(self.robot_model.lowerPositionLimit)):
            if np.isinf(self.robot_model.lowerPositionLimit[i]):
                self.robot_model.lowerPositionLimit[i] = 10
            if i >= (self.end_effector_index_list_joint[4] - 2 + 7):
                self.robot_model.lowerPositionLimit[i] = 0
        
        for i in range(len(self.robot_model.upperPositionLimit)):
            if np.isinf(self.robot_model.upperPositionLimit[i]):
                self.robot_model.upperPositionLimit[i] = 10
            if i >= (self.end_effector_index_list_joint[4] - 2 + 7):
                self.robot_model.upperPositionLimit[i] = 0
            
        lower_pos_lim = np.transpose(self.robot_model.lowerPositionLimit[np.newaxis])
        upper_pos_lim = np.transpose(self.robot_model.upperPositionLimit[np.newaxis])
        K_lim = np.identity(27)*2
        lower_pos_lim = np.dot(K_lim,(lower_pos_lim - np.transpose(self.current_joint_config[np.newaxis])))*(1/self.sampling_time)
        upper_pos_lim = np.dot(K_lim,(upper_pos_lim - np.transpose(self.current_joint_config[np.newaxis])))*(1/self.sampling_time)

        lower_pos_lim = np.delete(lower_pos_lim, 0, 0)
        upper_pos_lim = np.delete(upper_pos_lim, 0, 0)
        
        return lower_pos_lim, upper_pos_lim

    def comJacobian(self):
        J = pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.current_joint_config)
        self.comJ = np.dot(self.com_weight, J)
        return J

    def qpCartesianA(self):
        self.comJacobian()
        self.EndEffectorJacobians()
        self.TrunkJacobian()
        A = np.concatenate((self.end_effector_jacobians, self.comJ), axis=0)
        A = np.concatenate((A, self.trunkJ), axis=0)
        return A

    def cartesianTargetsEE(self, target_cartesian_pos, target_cartesian_vel):
        target_list = [0, 0, 0, 0, 0]
        if np.sum(target_cartesian_pos) == 0 and np.sum(target_cartesian_vel) == 0:
            self.cartesian_targetsEE = np.zeros(((self.n_of_EE*6),1))
        else:
            for i in range(len(self.end_effector_index_list_frame)):
                if np.sum(target_cartesian_pos[i]) == 0 and np.sum(target_cartesian_vel[i]) == 0:
                    target_list[i] = np.zeros((6,1))
                else:
                    rot = self.robot_data.oMf[self.end_effector_index_list_frame[i]].rotation
                    rot = self.Rot2Euler(rot)
                    rot = np.array([[0,0,0]]).T
                    x = target_cartesian_pos[i] - np.array([self.robot_data.oMf[self.end_effector_index_list_frame[i]].translation]).T
                    x = np.concatenate((x,rot),axis=0)
                    target_list[i] = target_cartesian_vel[i] + np.dot(self.EE_gain, x)
                    
            self.cartesian_targetsEE = target_list[0]
            for i in range(len(target_list)-1):
                self.cartesian_targetsEE = np.concatenate((self.cartesian_targetsEE,target_list[i+1]), axis=0)

    def cartesianTargetCoM(self, target_cartesian_pos, target_cartesian_vel):
        if np.sum(target_cartesian_pos) == 0 and np.sum(target_cartesian_vel) == 0:
            self.cartesian_targetsCoM = np.zeros((3,1))
        else:
            x = target_cartesian_pos - np.array([self.robot_data.com[0]]).T
            self.cartesian_targetsCoM = target_cartesian_vel + np.dot(self.com_gain, x)
            #print(self.cartesian_targetsCoM)
            #self.cartesian_targetsCoM = np.dot(K_cart, (target_cartesian_pos-np.array([self.robot_data.com[0]]).T))

    def cartesianTargetTrunk(self, target_cartesian_pos, target_cartesian_vel):
        if np.sum(target_cartesian_pos) == 0 and np.sum(target_cartesian_vel) == 0:
            self.cartesian_targetsTrunk = np.zeros((6,1))
        else:
            x = target_cartesian_pos - np.array([self.robot_data.oMf[self.trunk_frame_index].translation]).T
            rot = self.robot_data.oMf[self.trunk_frame_index].rotation
            rot = self.Rot2Euler(rot)
            #rot = np.array([[0,0,0]]).T
            x = np.concatenate((x,rot), axis=0)
            self.cartesian_targetsTrunk = target_cartesian_vel + np.dot(self.trunk_gain, x)


    def posAndVelTargetsCoM(self, objective):
        err = objective - np.array([self.robot_data.com[0]]).T
        print(err)
        step = err/(10000)
        base = np.array([self.robot_data.com[0]]).T
        planner_pos = [0]*10000
        planner_pos[0] = base
        for i in range(len(planner_pos)-1):
            planner_pos[i+1] = planner_pos[i] + step
        
        planner_vel = [0]*10000
        for i in range(len(planner_vel)):
            planner_vel[i] = 0 #step/5000
        
        return planner_pos, planner_vel
        
    def qpCartesianB(self, target_cartesian_pos_CoM, target_cartesian_vel_CoM, target_cartesian_pos_EE, target_cartesian_vel_EE, target_cartesian_pos_trunk, target_cartesian_vel_trunk):
        self.cartesianTargetCoM(target_cartesian_pos_CoM, target_cartesian_vel_CoM)
        self.cartesianTargetsEE(target_cartesian_pos_EE, target_cartesian_vel_EE)
        self.cartesianTargetTrunk(target_cartesian_pos_trunk, target_cartesian_vel_trunk)
        b = np.concatenate((self.cartesian_targetsEE ,self.cartesian_targetsCoM), axis=0)
        b = np.concatenate((b, self.cartesian_targetsTrunk), axis=0)
        return b

    #Initial Configuration
    #def InitialConfig(self):
        

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
