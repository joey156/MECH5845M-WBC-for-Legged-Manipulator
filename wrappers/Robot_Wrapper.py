import pinocchio as pin
import numpy as np
import math
import time
from QP_Wrapper import QP
from klampt.model import trajectory
from PID_Controller import PID

large_width = 400
np.set_printoptions(linewidth=large_width)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

class RobotModel:
    def __init__(self, urdf_path, EE_frame_names, EE_joint_names, G_base, imu, FR_hip_joint):
        #initialise pinocchio model and data
        self.robot_model = pin.buildModelFromUrdf(urdf_path)
        self.robot_data = self.robot_model.createData()
        self.joint_names = self.robot_model.names
        self.no_DoF = self.robot_model.nv
        self.no_config = self.robot_model.nq
        #set robot to a neutral stance and initalise parameters
        #self.stand_joint_config = np.array([0,0,0,0,0,0,1,0.037199,0.660252,-1.200187,-0.028954,0.618814,-1.183148,0.048225,0.690008,-1.254787,-0.050525,0.661355,-1.243304, 0, -1.6, 1.6, 0, 0, 0, 0.02, -0.02])
        #self.stand_joint_config = np.array([ 0.,-0.,0.,0.,0.,0.,1.,0.,0.625,-0.754,-0.,0.625,-0.754,0.,0.625,-0.754,-0.,0.625,-0.754,0.,0.068,0.287,-0.36,-0.,0.,0.,0.])
        self.stand_joint_config = np.array([ 0., 0., 0., -0.001, 0.012, -0.001, 1., 0.007, 0.868, -1.168, -0.001, 0.869, -1.169, 0.007, 0.823, -1.094, -0.001, 0.823, -1.094, -0.002, -2.133, 0.945, 1.112, 0.002, 0., 0., 0.])
        self.current_joint_config = 0
        self.EE_frame_names = EE_frame_names
        self.EE_joint_names = EE_joint_names
        self.arm_base_id = self.robot_model.getJointId(G_base) #G_base
        self.FR_hip_joint = self.robot_model.getJointId(FR_hip_joint)
        self.n_velocity_dimensions = self.robot_model.nv
        self.n_configuration_dimensions = self.robot_model.nq
        print(self.n_velocity_dimensions)
        print(self.n_configuration_dimensions)
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

        self.trunk_frame_index= self.robot_model.getFrameId(imu, pin.FIXED_JOINT)

        self.count = 0

        # find joint task parameters
        self.jointTaskA = self.qpJointA()

        # cartesian task weights
        self.com_weight = np.identity(3) * 1 # 20#1.2
        self.trunk_weight = np.identity(6) * 300 #15#1
        self.FL_weight = np.identity(6) * 40 #20#0.8 18 for wx100
        self.FR_weight = np.identity(6) * 40 #20#0.8 18 for wx100
        self.RL_weight = np.identity(6) * 40 #20#0.8 18 for wx100
        self.RR_weight = np.identity(6) * 40 #20#0.8 18 for wx100
        self.grip_weight = np.identity(6) * 50 #15#1
        self.EE_weight = [self.FL_weight, self.FR_weight, self.RL_weight, self.RR_weight, self.grip_weight]

        # task weights
        self.taskWeightCart = 6
        self.taskWeightJoint = 0.05

        # identify which tasks are active
        self.taskActiveEE = False
        self.taskActiveCoM = False
        self.taskActiveTrunk = False
        self.taskActiveJoint = False

        #sampling time parameters
        self.previous_time = 0
        self.sampling_time = 0.002 #in seconds (2ms)
        self.dt = 0.002

        # cartesian proportional gains
        self.com_gain = np.identity(3)* 1 #1.5#1.631
        self.trunk_gain = np.identity(6)* 0.001 #0.5425
        self.FL_gain = np.identity(6) * 0.001
        self.FR_gain = np.identity(6) * 0.001
        self.RL_gain = np.identity(6) * 0.001
        self.RR_gain = np.identity(6) * 0.001
        self.GRIP_gain = np.identity(6) * 0.1
        self.EE_gains = [self.FL_gain, self.FR_gain, self.RL_gain, self.RR_gain, self.GRIP_gain]
        
        self.comJacobian()
        self.cartesian_targetsEE = 0
        self.cartesian_targetsCoM = 0
        self.end_effector_jacobians = 0
        self.cartesian_targetsTrunk = 0
        self.default_trunk_ori = np.array([[0,0,0]]).T
        self.default_EE_ori_list = [np.array([[0,0,0]]).T, np.array([[0,0,0]]).T, np.array([[0,0,0]]).T, np.array([[0,0,0]]).T, np.array([[0,0,0]]).T]
        self.firstQP = True
        self.qp = None
        self.FL_base_pos = self.robot_data.oMf[self.end_effector_index_list_frame[1]].translation
        self.print_ = False
        self.setInitialState()
        self.print_ = True
        self.FL_base_pos = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[1]].translation)

        # IMU PID controller gains
        self.IMU_Kp = 0.9
        self.IMU_Ki = 0.1 #0.9
        self.IMU_Kd = 0 #0.005
        # initialise IMU PID
        self.IMU_setpoint = self.current_joint_config[:7]
        self.IMU_PID = PID(self.IMU_Kp, self.sampling_time, self.IMU_setpoint, self.IMU_Ki, self.IMU_Kd)


    def setTasks(self, EE=False, CoM=False, Trunk=False, Joint=False):
        self.taskActiveEE = EE
        self.taskActiveCoM = CoM
        self.taskActiveTrunk = Trunk
        self.taskActiveJoint = Joint


    def setInitialState(self):
        
        # set a neutral configuration
        q = pin.neutral(self.robot_model)
        print(q.shape)
        self.updateState(q, feedback=False)
        # find initial cartesian position of end effectors and trunk
        EE_pos_FL = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[0]].translation]).T
        EE_pos_FR = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[1]].translation]).T 
        EE_pos_RL = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[2]].translation]).T
        EE_pos_RR = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[3]].translation]).T 
        EE_pos_GRIP = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[4]].translation]).T
        Trunk_target_pos = np.array([self.robot_data.oMf[self.trunk_frame_index].translation]).T
        print("trunk pos", Trunk_target_pos)
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
        multiplier_F[2,2] = 0.8 #0.65
        multiplier_F[0,0] = 0.65
        multiplier_R[2,2] = 0.8 #0.65
        multiplier_R[0,0] = 1.3
        multiplier_G[2,2] = 2 #2.8 #0.7
        

        EE_G_pos_2 = EE_pos_GRIP.reshape((3,)).tolist()
        print(EE_G_pos_2)
        EE_G_pos_2[2] = self.robot_data.oMi[self.arm_base_id].translation[2]
        EE_G_pos_2[0] = self.robot_data.oMi[self.FR_hip_joint].translation[0]#self.robot_data.oMi[self.arm_base_id+5].translation[2]
        #EE_G_pos_2[1] = 1
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

        # select the tasks that are active
        self.setTasks(EE=True, CoM=False, Trunk=True, Joint=True)
        
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
            A = self.qpA()
            b = self.qpb(com_pos, com_vel, EE_target_pos, EE_target_vel, Trunk_target_pos, Trunk_target_vel).reshape((A.shape[0],))
            #print(b)
            # solver QP
            qp = QP(A, b, lb, ub, n_of_velocity_dimensions=self.n_velocity_dimensions)
            qp_vel = qp.solveQP()

            #print("EE_taget_pos: ", EE_target_pos[0].reshape((1,3)), EE_target_pos[1].reshape((1,3)), EE_target_pos[2].reshape((1,3)))
            #print("qpvel: ", qp_vel)

            # find the new joint angles from the QP optimised joint velocities
            self.jointVelocitiestoConfig(qp_vel, True)

            
        print("Initial state set successfully")

        for i in range(len(self.current_joint_config)):
            if i < 6:
                self.current_joint_config[i] = 0

        joint_config = self.current_joint_config
        self.updateState(joint_config, feedback=False)
            
                
    def updateState(self, joint_config, base_config=0, feedback=True): # put floatig base info here
        if feedback == True:
            #base_config = self.IMU_PID.PIDUpdate(base_config,self.current_joint_config[:7])
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
        #print("joint_config",joint_config)

        self.default_trunk_ori = self.Rot2Euler(self.robot_data.oMf[self.trunk_frame_index].rotation)
        self.default_EE_ori_list[0] = self.Rot2Euler(self.robot_data.oMf[self.end_effector_index_list_frame[0]].rotation)
        self.default_EE_ori_list[1] = self.Rot2Euler(self.robot_data.oMf[self.end_effector_index_list_frame[1]].rotation)
        self.default_EE_ori_list[2] = self.Rot2Euler(self.robot_data.oMf[self.end_effector_index_list_frame[2]].rotation)
        self.default_EE_ori_list[3] = self.Rot2Euler(self.robot_data.oMf[self.end_effector_index_list_frame[3]].rotation)
        self.default_EE_ori_list[4] = self.Rot2Euler(self.robot_data.oMf[self.end_effector_index_list_frame[4]].rotation)
        
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
        J = pin.getFrameJacobian(self.robot_model, self.robot_data, self.trunk_frame_index, pin.LOCAL_WORLD_ALIGNED)
        self.trunkJ = np.dot(self.trunk_weight, J)
        
            
    def jointVelLimitsArray(self, initial_config=False): # returns an array for the upper and lower joint velocity limits which will be used for QP
        vel_lim = self.robot_model.velocityLimit
        
        for i in range(len(vel_lim)):
            if (np.isinf(vel_lim[i])):
                vel_lim[i] = 100
            if i >= (self.end_effector_index_list_joint[4] - 2 + 6):
                vel_lim[i] = 0
 
            
        lower_vel_lim = -vel_lim[np.newaxis]
        upper_vel_lim = vel_lim[np.newaxis]
        return lower_vel_lim, upper_vel_lim

    def jointPosLimitsArray(self, initial_config=False): # returns an array for the upper and lower joint position limits, these have been turned into velocity limits
        for i in range(len(self.robot_model.lowerPositionLimit)):
            if np.isinf(self.robot_model.lowerPositionLimit[i]):
                self.robot_model.lowerPositionLimit[i] = 100
            if i >= (self.end_effector_index_list_joint[4] - 2 + 7):
                self.robot_model.lowerPositionLimit[i] = 0
        
        for i in range(len(self.robot_model.upperPositionLimit)):
            if np.isinf(self.robot_model.upperPositionLimit[i]):
                self.robot_model.upperPositionLimit[i] = 10
            if i >= (self.end_effector_index_list_joint[4] - 2 + 7):
                self.robot_model.upperPositionLimit[i] = 0
            
        lower_pos_lim = np.transpose(self.robot_model.lowerPositionLimit[np.newaxis])
        upper_pos_lim = np.transpose(self.robot_model.upperPositionLimit[np.newaxis])
        K_lim = np.identity(self.n_configuration_dimensions)*0.5
        lower_pos_lim = np.dot(K_lim,(lower_pos_lim - np.transpose(self.current_joint_config[np.newaxis])))*(1/self.sampling_time)
        upper_pos_lim = np.dot(K_lim,(upper_pos_lim - np.transpose(self.current_joint_config[np.newaxis])))*(1/self.sampling_time)

        lower_pos_lim = np.delete(lower_pos_lim, 0, 0)
        upper_pos_lim = np.delete(upper_pos_lim, 0, 0)
        
        return lower_pos_lim, upper_pos_lim

    def footConstraint(self):
        C = pin.getFrameJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_frame[0], pin.LOCAL_WORLD_ALIGNED)[:3]
        fill = np.zeros((3, self.n_velocity_dimensions))
        C = np.concatenate((C, fill), axis=0)
        
        for i in range(len(self.end_effector_index_list_frame)-2):
            Jtmp = pin.getFrameJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_frame[i+1], pin.LOCAL_WORLD_ALIGNED)[:3]
            Jtmp = np.concatenate((Jtmp, fill), axis=0)
            C = np.concatenate((C, Jtmp), axis=0)
        C = np.concatenate((C, np.zeros((6, self.n_velocity_dimensions))), axis=0)
        Clb = np.zeros(C.shape[0]).reshape((C.shape[0],))
        Cub = np.zeros(C.shape[0]).reshape((C.shape[0],))
        
        return C, Clb, Cub


    def CoMConstraint(self):
        C= pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.current_joint_config)[:2]
        C = np.concatenate((C, np.zeros((1, self.n_velocity_dimensions))), axis=0)
        Cub = self.robot_data.oMf[self.end_effector_index_list_frame[0]].translation.reshape((C.shape[0],)) # FL
        Clb = self.robot_data.oMf[self.end_effector_index_list_frame[3]].translation.reshape((C.shape[0],)) # RR
        return C, Clb, Cub

    def findConstraints(self):
        C, Clb, Cub = self.footConstraint()
        D, Dlb, Dub = self.CoMConstraint()

        C = np.concatenate((C, D), axis=0)
        Clb = np.concatenate((Clb, Dlb), axis=0).reshape((C.shape[0],))
        Cub = np.concatenate((Cub, Dub), axis=0).reshape((C.shape[0],))

        return C.T, Clb, Cub

        
    def comJacobian(self):
        J = pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.current_joint_config)
        self.comJ = np.dot(self.com_weight, J)
        return J

    def qpCartesianA(self):

        # this list will contain the jacobians from active tasks
        jacobian_list = []

        # find the jacobains for the active tasks
        if self.taskActiveEE == True:
            self.EndEffectorJacobians()
            jacobian_list.append(self.end_effector_jacobians)

        if self.taskActiveCoM == True:
            self.comJacobian()
            jacobian_list.append(self.comJ)

        if self.taskActiveTrunk == True:
            self.TrunkJacobian()
            jacobian_list.append(self.trunkJ)

        """
        A = np.concatenate((self.end_effector_jacobians, self.comJ), axis=0)
        A = np.concatenate((A, self.trunkJ), axis=0)
        """

        # combine all jacobians to find A
        for i in range(len(jacobian_list)):
            if i == 0:
                A = jacobian_list[i]
            else:
                A = np.concatenate((A, jacobian_list[i]), axis=0)
        
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
                    target_list[i] = self.calcTargetVel(target_cartesian_pos[i], self.default_EE_ori_list[i], self.end_effector_index_list_frame[i], self.EE_gains[i])

                    
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
            self.cartesian_targetsTrunk = self.calcTargetVel(target_cartesian_pos, self.default_trunk_ori, self.trunk_frame_index, self.trunk_gain)
        else:
            self.cartesian_targetsTrunk = self.calcTargetVel(target_cartesian_pos, self.default_trunk_ori, self.trunk_frame_index, self.trunk_gain)


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


    def calcTargetVel(self, target_pos, target_rot, frame_id, gain):
        # find the current cartisian position of the end effector
        fk_pos = np.array([self.robot_data.oMf[frame_id].translation]).T
        fk_ori = np.array([self.robot_data.oMf[frame_id].rotation]).T
        fk_ori = self.Rot2Euler(fk_ori)
        fk_cart = np.concatenate((fk_pos, fk_ori), axis=0)
        #fk_cart = self.targetWorldToLocal(fk_cart)

        # find desired cartesian position
        des_cart = np.concatenate((target_pos, target_rot), axis=0)
        des_cart = self.targetWorldToLocal(des_cart)

        # calculate desired velocity
        pos_vel = (des_cart[:3] - fk_cart[:3]) / self.dt
        ori_vel = np.array([[0,0,0]]).T
        des_vel = np.concatenate((pos_vel,ori_vel), axis=0)
        
        # calculate target end effector velocity
        target_vel = des_vel + np.dot(gain, (des_cart - fk_cart))

        return target_vel

    def targetWorldToLocal(self, world_target):
        ori_offset = self.Rot2Euler(self.robot_data.oMf[3].rotation)
        pos_offset = (self.robot_data.oMf[self.end_effector_index_list_frame[1]].translation - self.FL_base_pos).reshape(ori_offset.shape)
        if self.print_ == True:
            if self.count % 5 == 0:
                x = 1
                #print("new:", self.robot_data.oMf[self.end_effector_index_list_frame[1]].translation)
                #print("old:", self.FL_base_pos)
                #print(pos_offset.T)
            self.count = self.count + 1
        offset = np.vstack((pos_offset, ori_offset))
        #print("pin offset:", offset.T)
        local_target = world_target - offset
        return local_target

        
    def qpCartesianB(self, target_cartesian_pos_CoM, target_cartesian_vel_CoM, target_cartesian_pos_EE, target_cartesian_vel_EE, target_cartesian_pos_trunk, target_cartesian_vel_trunk):
        # this list will contain the targets from active tasks
        target_list = []

        # for the active tasks find their targets
        if self.taskActiveEE == True:
            self.cartesianTargetsEE(target_cartesian_pos_EE, target_cartesian_vel_EE)
            target_list.append(self.cartesian_targetsEE)

        if self.taskActiveCoM == True:
            self.cartesianTargetCoM(target_cartesian_pos_CoM, target_cartesian_vel_CoM)
            target_list.append(self.cartesian_targetsCoM)

        if self.taskActiveTrunk == True:
            self.cartesianTargetTrunk(target_cartesian_pos_trunk, target_cartesian_vel_trunk)
            target_list.append(self.cartesian_targetsTrunk)

        # combine all targets to find b
        for i in range(len(target_list)):
            if i == 0:
                b = target_list[i]
            else:
                b = np.concatenate((b, target_list[i]), axis=0)

        """
        b = np.concatenate((self.cartesian_targetsEE ,self.cartesian_targetsCoM), axis=0)
        b = np.concatenate((b, self.cartesian_targetsTrunk), axis=0)
        """

        return b


    def qpJointA(self):

        U = np.identity(self.n_velocity_dimensions)
        a = np.ones(self.n_velocity_dimensions)*(1/self.n_velocity_dimensions)
        A = U*a

        return A


    def qpJointb(self):

        # Tikhonov Regularization (default
        if self.taskActiveJoint == True:
            u = np.zeros((self.n_velocity_dimensions, 1))

        # eleminate high frequency oscillations
        if self.taskActiveJoint == "PREV":
            u = np.delete(self.current_joint_config, 6).reshape((self.n_velocity_dimensions, 1))

        # manipulability gradient to reduce singularity risk
        if self.taskActiveJoint == "MANI":
            u = []
            q = np.copy(self.current_joint_config)
            deltaq = 0.0002
            
            for i in range(self.n_velocity_dimensions):
                if i < 6:
                    joint_id = 1
                else:
                    joint_id = i + 1 - 5
                    
                q[i] = q[i] + deltaq
                self.updateState(q, feedback=False)
                J = pin.getJointJacobian(self.robot_model, self.robot_data, joint_id, pin.LOCAL_WORLD_ALIGNED)
                f1 = math.sqrt(np.linalg.det(np.dot(J, J.T)))
                q[i] = q[i] - (deltaq * 2)
                self.updateState(q, feedback=False)
                J = pin.getJointJacobian(self.robot_model, self.robot_data, joint_id, pin.LOCAL_WORLD_ALIGNED)
                f2 = math.sqrt(np.linalg.det(np.dot(J, J.T)))
                u.append(0.5*(f1-f2)/deltaq)

            u = np.array(u).reshape((self.n_velocity_dimensions, 1))
            self.updateState(self.current_joint_config, feedback=False)

        a = np.ones((self.n_velocity_dimensions, 1))*(1/self.n_velocity_dimensions)

        b = a*u

        return b



    def qpA(self):

        A = self.qpCartesianA() * self.taskWeightCart

        if self.taskActiveJoint == True or self.taskActiveJoint == "PREV":
            A = np.concatenate((A, (self.jointTaskA * self.taskWeightJoint)), axis=0)

        return A


    def qpb(self, target_cartesian_pos_CoM, target_cartesian_vel_CoM, target_cartesian_pos_EE, target_cartesian_vel_EE, target_cartesian_pos_trunk, target_cartesian_vel_trunk):
        
        b = self.qpCartesianB(target_cartesian_pos_CoM, target_cartesian_vel_CoM, target_cartesian_pos_EE, target_cartesian_vel_EE, target_cartesian_pos_trunk, target_cartesian_vel_trunk)
        b = b * self.taskWeightJoint
        
        if self.taskActiveJoint == True or self.taskActiveJoint == "PREV":

            b = np.concatenate((b, (self.qpJointb() * self.taskWeightJoint)), axis=0)
            
        return b


    def runWBC(self, base_config, target_cartesian_pos_CoM=None, target_cartesian_vel_CoM=None, target_cartesian_pos_EE=None, target_cartesian_vel_EE=None, target_cartesian_pos_trunk=None, target_cartesian_vel_trunk=None):
        # find joint position and velocity limits
        #print(self.current_joint_config)

        # ensure sample time is maintained
        while ((time.time() - self.previous_time) < self.sampling_time):
            self.dt = time.time() - self.previous_time

        # update the time the WBC ran
        self.previous_time = time.time()
        
        lower_vel_lim, upper_vel_lim = self.jointVelLimitsArray()
        lower_pos_lim, upper_pos_lim = self.jointPosLimitsArray()
        lb = np.concatenate((lower_vel_lim.T, lower_pos_lim), axis=0).reshape(((self.n_velocity_dimensions*2),))
        ub = np.concatenate((upper_vel_lim.T, upper_pos_lim), axis=0).reshape(((self.n_velocity_dimensions*2),))

        #print("lb:", lb)
        #print("ub:", ub)

        # find cartesian tasks (A and b)
        A = self.qpA()
        b = self.qpb(target_cartesian_pos_CoM, target_cartesian_vel_CoM, target_cartesian_pos_EE, target_cartesian_vel_EE, target_cartesian_pos_trunk, target_cartesian_vel_trunk).reshape((A.shape[0],))

        # find foot constraints
        C, Clb, Cub = self.findConstraints()

        # solve qp
        if self.firstQP == True:
            self.qp = QP(A, b, lb, ub, C, Clb, Cub, n_of_velocity_dimensions=self.n_velocity_dimensions)
            q_vel = self.qp.solveQP()
            self.firstQP = False
        else:
            q_vel = self.qp.solveQPHotstart(A, b, lb, ub, C, Clb, Cub)

        # find the new joint angles from the solved joint velocities
        joint_config = self.jointVelocitiestoConfig(q_vel, False)[7:]

        #self.jointVelocitiestoConfig(q_vel, True)

        # update robot model with new joint and base configuration
        self.updateState(joint_config, base_config)

        # return the new joint configuration

        FL_leg = joint_config[0:3]
        FR_leg = joint_config[3:6]
        RL_leg = joint_config[6:9]
        RR_leg = joint_config[9:12]
        grip = joint_config[12:]
        
        return FL_leg, FR_leg, RL_leg, RR_leg, grip       
        

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
