import pinocchio as pin
from scipy.spatial.transform import Rotation as R
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
    def __init__(self, urdf_path, mesh_dir_path, EE_frame_names, EE_joint_names, G_base, imu, FR_hip_joint, hip_waist_joint_names, foot_offset=False):
        """ Initialise Pinocchio model, data and geometry model """
        self.robot_model= pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.geom_model = pin.buildGeomFromUrdf(self.robot_model, urdf_path, mesh_dir_path, pin.GeometryType.COLLISION)
        self.robot_data = self.robot_model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)
        self.joint_names = self.robot_model.names
        self.foot_radius = 0

        """ Find frame IDs """
        # trunk frame
        self.trunk_frame_index = self.robot_model.getFrameId(imu, pin.FIXED_JOINT)
        
        # end-effector names
        self.current_joint_config = 0
        self.EE_frame_names = EE_frame_names
        self.EE_joint_names = EE_joint_names
        self.hip_waist_joint_names = hip_waist_joint_names
        self.arm_base_id = self.robot_model.getJointId(G_base)
        self.arm_base_frame_id = self.robot_model.getFrameId(G_base, pin.JOINT)
        self.FR_hip_joint = self.robot_model.getJointId(FR_hip_joint)
        self.n_velocity_dimensions = self.robot_model.nv
        self.n_configuration_dimensions = self.robot_model.nq
        self.n_of_EE = 5
        self.end_effector_index_list_frame = [] 
        self.end_effector_index_list_joint = []
        self.hip_waist_joint_index_list_frame = []
        for i in range(len(self.EE_joint_names)):
            ID = self.robot_model.getFrameId(self.EE_frame_names[i], pin.FIXED_JOINT)
            self.end_effector_index_list_frame.append(ID)
            ID = self.robot_model.getJointId(self.EE_joint_names[i])
            self.end_effector_index_list_joint.append(ID)
            ID = self.robot_model.getFrameId(self.hip_waist_joint_names[i], pin.JOINT)
            self.hip_waist_joint_index_list_frame.append(ID)

        """ Find foot offset if enabled """
        if foot_offset == True:
            foot_geom_name = self.geom_model.geometryObjects[self.end_effector_index_list_joint[0]+1].name
            foot_geom_id = self.geom_model.getGeometryId(foot_geom_name)
            self.foot_radius = self.geom_model.geometryObjects[8].geometry.radius

        """ Find arm range """
        arm_base_placement = np.copy(self.robot_data.oMf[self.arm_base_frame_id].translation)
        gripper_placement = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[4]].translation)
        self.arm_reach = np.sum(gripper_placement - arm_base_placement)
        
        """ Set weights """
        # cartesian task DoF weighting matrix
        self.trunk_weight = np.identity(6) * 125
        self.FL_weight = np.identity(6) * 4 
        self.FR_weight = np.identity(6) * 4
        self.RL_weight = np.identity(6) * 4 
        self.RR_weight = np.identity(6) * 4 
        self.grip_weight = np.identity(6) * 120 * self.arm_reach
        self.EE_weight = [self.FL_weight, self.FR_weight, self.RL_weight, self.RR_weight, self.grip_weight]
        self.DoF_W = np.zeros((6*len(self.end_effector_index_list_frame), 6*len(self.end_effector_index_list_frame)))
        for i in range(len(self.end_effector_index_list_frame)):
            self.DoF_W[i*6:(i+1)*6, i*6:(i+1)*6] = self.EE_weight[i]

        # task weights
        self.cart_task_weight_FR = 1
        self.cart_task_weight_FL = 1
        self.cart_task_weight_RR = 1
        self.cart_task_weight_RL = 1
        self.cart_task_weight_GRIP = 1
        self.cart_task_weight_Trunk = 1
        self.cart_task_weight_EE_list = [self.cart_task_weight_FR, self.cart_task_weight_FL, self.cart_task_weight_RR, self.cart_task_weight_RL, self.cart_task_weight_GRIP]
        self.joint_task_weight = 1

        # identify which tasks are active
        self.task_active_EE = False
        self.task_active_Trunk = False
        self.task_active_Joint = False

        # step time parameters
        self.previous_time = 0
        self.step_time = 0.002 # 2ms
        self.dt = 0.002

        # Cartesian task proportional gains
        self.trunk_gain = np.identity(6)* 0.5
        self.FL_gain = np.identity(6) * 0.5
        self.FR_gain = np.identity(6) * 0.5
        self.RL_gain = np.identity(6) * 0.5
        self.RR_gain = np.identity(6) * 0.5
        self.GRIP_gain = np.identity(6) * 0.5
        self.EE_gains = [self.FL_gain, self.FR_gain, self.RL_gain, self.RR_gain, self.GRIP_gain]

        """ Set peliminary statuses, targets and values """
        # set initial targets
        self.default_trunk_ori = np.array([[0,0,0]]).T
        self.default_EE_ori_list = [np.array([[0,0,0]]).T, np.array([[0,0,0]]).T, np.array([[0,0,0]]).T, np.array([[0,0,0]]).T, np.array([[0,0,0]]).T]#
        
        # set initial trunk velocity
        self.prev_trunk_ref = np.array([0,0,0])
        self.old_ref_trunk_rot_matrix = np.zeros((3,3))

        # set initial EE velocities
        self.prev_EE_pos = [0,0,0,0,0]

        # set initial EE_CoM_rot
        self.prev_EE_CoM_rot = [0,0,0,0,0]

        # initialse list to store frame placements
        self.EE_frame_pos = [0,0,0,0,0]
        self.trunk_frame_pos = 0

        # set initial status variables
        self.firstQP = True
        self.qp = None
        self.FL_base_pos = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[1]].translation)
        self.print_ = False
        self.initialised = False
        self.end_effector_A = 0
        self.end_effector_B = 0
        self.trunk_A = 0
        self.trunk_B = 0

        """ Set initail state """
        self.setInitailState()
        self.initialised = True
        self.FL_base_pos = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[1]].translation)
        self.trunk_base_pos = np.copy(self.robot_data.oMf[self.trunk_frame_index].translation)
        self.prev_trunk_pos = np.copy(self.robot_data.oMf[self.trunk_frame_index].translation)


    def setTasks(self, EE=False, Trunk=False, Joint=False):
        self.task_active_EE = EE
        self.task_active_Trunk = Trunk
        self.task_active_Joint = Joint


    def setInitialState(self):

        # set a neutral configuration
        q = pin.neutral(self.robot_model)
        self.updateState(q, feedback=False)

        # set default EE and trunk orientation
        self.default_trunk_ori = R.from_matrix(self.robot_data.oMf[self.trunk_frame_index].rotation)
        self.default_trunk_ori = self.default_trunk_ori.as_euler('xyz').reshape(3,1)
        for i in range(len(self.default_EE_ori_list)):
            self.default_EE_ori_list[i] = R.from_matrix(self.robot_data.oMf[self.end_effector_index_list_frame[i]].rotation)
            self.default_EE_ori_list[i] = self.default_EE_ori_list[i].as_euler('xyz').reshape(3,1)

        # set initial target positions
        Trunk_target_pos = self.trunk_frame_pos.T
        EE_target_pos = [self.EE_frame_pos[0].T, self.EE_frame_pos[1].T, self.EE_frame_pos[2].T, self.EE_frame_pos[3].T, self.EE_frame_pos[4].T]

        """ Setting up trajectories to desired initial configuration """
        multiplier_F = np.identity(3)
        multiplier_R = np.identity(3)
        multiplier_G = np.identity(3)
        multiplier_F[2,2] = 0.7 #0.65
        multiplier_F[0,0] = 0.8
        multiplier_R[2,2] = 0.7 #0.65
        multiplier_R[0,0] = 1.1
        multiplier_G[2,2] = 2.5 #2.8 #0.7
        EE_G_pos_2 = EE_target_pos[4].reshape((3,)).tolist()
        EE_G_pos_2[2] = self.robot_data.oMi[self.arm_base_id].translation[2]
        EE_G_pos_2[0] = self.robot_data.oMi[self.FR_hip_joint].translation[0]
        
        # set trajecotry milestones
        EE_FR_milestones = [EE_target_pos[0].reshape((3,)).tolist(), np.dot(EE_target_pos[0].reshape((3,)), multiplier_F).tolist()]
        EE_FL_milestones = [EE_target_pos[1].reshape((3,)).tolist(), np.dot(EE_target_pos[1].reshape((3,)), multiplier_F).tolist()]
        EE_RR_milestones = [EE_target_pos[2].reshape((3,)).tolist(), np.dot(EE_target_pos[2].reshape((3,)), multiplier_R).tolist()]
        EE_RL_milestones = [EE_target_pos[3].reshape((3,)).tolist(), np.dot(EE_target_pos[3].reshape((3,)), multiplier_R).tolist()]
        EE_G_milestones = [EE_target_pos[4].reshape((3,)).tolist(), np.dot(EE_G_pos_2, multiplier_G).tolist()]
        
        # setting up the trajectories for the feed end effectors
        EE_FL_traj = trajectory.Trajectory(milestones=EE_FL_milestones)
        EE_FR_traj = trajectory.Trajectory(milestones=EE_FR_milestones)
        EE_RL_traj = trajectory.Trajectory(milestones=EE_RL_milestones)
        EE_RR_traj = trajectory.Trajectory(milestones=EE_RR_milestones)
        EE_G_traj = trajectory.Trajectory(milestones=EE_G_milestones)
        EE_traj = [EE_FL_traj, EE_FR_traj, EE_RL_traj, EE_RR_traj, EE_G_traj]

        # select the tasks that are active
        self.setTasks(EE=True, Trunk=True, Joint=False)

        # set the trajectory interval
        trajectory_interval = np.arange(0,len(EE_FR_milestones), 0.001).tolist()

        """ Solve QP until desired initial configuration is reached """
        for i in trajectory_interval:

            # set new desired position for each foot
            for ii in range(len(EE_traj)):
                target_pos = EE_traj[ii]
                EE_target_pos[ii] = np.array(target_pos.eval(i)).reshape(3,1)

            # store feet target positions
            self.FR_target_cartesian_pos = EE_target_pos[0]
            self.FL_target_cartesian_pos = EE_target_pos[1]
            self.RR_target_cartesian_pos = EE_target_pos[2]
            self.RL_target_cartesian_pos = EE_target_pos[3]

            # find joint limits
            lower_vel_lim, upper_vel_lim = self.jointVelLimitsArray(True)
            lower_pos_lim, upper_pos_lim = self.jointPosLimitsArray(True)
            lb = np.concatenate((lower_vel_lim.T, lower_pos_lim), axis=0).reshape(((self.n_velocity_dimensions*2),))
            ub = np.concatenate((upper_vel_lim.T, upper_pos_lim), axis=0).reshape(((self.n_velocity_dimensions*2),))

            # find A and b for QP
            A = self.qpA()
            b = self.qpb(EE_target_pos, Trunk_target_pos).reshape((A.shape[0],))

            # solve QP
            qp = QP(A, b, lb, ub, n_of_velocity_dimensions=self.n_velocity_dimensions)
            q_vel = qp.solveQP()

            # find the new joint angles from teh QP optimised joint velocities
            self.jointVelociiestoConfig(q_vel, True)

        # reset base orientation
        for i in range(len(self.current_joint_config)):
            if i < 6 and i > 2:
                self.current _joint_config[i] = 0

        joint_config = self.current_joint_config
        self.updateState(joint_config, feedback=False)

        # find and set world position of the trunk
        height_offset = (self.EE_frame_pos[0][2], self.EE_frame_pos[1][2], self.EE_frame_pos[2][2], self.EE_frame_pos[3][2])/4
        joint_config[2] = height_offset + self.foot_radius
        self.updateState(joint_config, feedback=False)
        joint_config = self.current_jointconfig[7:]
        
        self.FL_leg = joint_config[0:3]
        self.FR_leg = joint_config[3:6]
        self.RL_leg = joint_config[6:9]
        self.RR_leg = joint_config[9:12]
        self.grip = joint_config[12:]

        self.fristQP = False

        print("Initial state set successfully")



    def updateState(self, joint_config, base_config=0, feedback=True, running=False):
        if feedback == True and running == True:
            config = np.concatenate((base_config, joint_config), axis-0)

        else:
            config = joint_config

        # update robot state
        pin.forwardKinematics(self.robot_model, self.robot_data, config)
        self.previous_joint_config = self.current_joint_config
        self.current_joint_config = config
        self.J = pin.computeJointJacobians(self.robot_model, self.robot_data, config)
        pin.framesForwardKinematics(self.robot_model, self.robot_data, config)
        pin.updateFramePlacements(self.robot_model, self,robot_data)

        # store frame placements
        self.trunk_frame_pos = np.copy(self.robot_data.oMf[self.trunk_frame_index].translation)
        for i in range(len(self.EE_frame_pos)):
            self.EE_frame_pos[i] = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[i]].translation)

        if running == True:
            # estimate base translation
            base_pos = self.trunkWorldPos()
            config = np.concatenate((base_pos, self.current_joint_config[3:]), axis=0)

            # update robot state
            pin.forwardKinematics(self,.robot_model, self.robot_data, config)
            self.previous_joint_config = self.current_joint_config
            self.current_joint_config = config
            self.J = pin.computeJointJacobians(self.robot_model, self.robot_data, config)
            pin.framesForwardKinematics(self.robot_model, self.robot_data, config)
            pin.updateFramePlacements(self.robot_model, self.robot_data)

            # store frame placements
            self.trunk_frame_pos = np.copy(self.robot_data.oMf[self.trunk_frame_index].translation)
            for i in range(len(self.EE_frame_pos)):
                self.EE_frame_pos[i] = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[i]].translation)

        # update default EE desired orientation
        for i in range(len(self.default_EE_ori_list)):
            self.default_EE_ori_list[i] = R.from_matrix(self.robot_data.oMf[self.end_effector_index_list_frame[i]].rotation)
            self.default_EE_ori_list[i] = self.default_EE_ori_list[i].as_euler('xyz').reshape(3,1)


    def jointVelocitiestoConfig(self, joint_vel, update_model=False):
        new_config = pin.integrate(self.robot_model, self.current_joint_config, joint_vel)
        if update_model == True:
            self.updateState(new_config, feedback=False, running=False)
        if update_model == False:
            return new_config
        

    def endEffectorA(self):
        if self.initialised == True:
            frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        else:
            frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            
        self.end_effector_A = pin.getFrameJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_frame[0], frame).T
        self.end_effector_A = self.end_effector_A * self.cart_task_weight_EE_list[0]
        
        for i in range(len(self.end_effector_index_list_frame)-1):
            if i < 3:
                J = pin.getFrameJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_frame[i+1], frame).T
                J = J * self.cart_task_weight_EE_list[i+1]
            else:
                J = pin.getFrameJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_frame[i+1], frame).T
                J = J * self.cart_task_weight_EE_list[i+1]
            self.end_effector_A = np.concatenate((self.end_effector_A, J), axis = 1)

        self.end_effector_A = np.dot(self.DoF_W, self.end_effector_A.T)


    def trunkA(self):
        self.trunk_A = pin.getFrameJacobian(self.robot_model, self.robot_data, self.trunk_frame_index, pin.ReferenceFrame.WORLD)
        self.trunk_A = self.trunk_A * self.cart_task_weight_Trunk


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
            if i < 7:
                self.robot_model.lowerPositionLimit[i] = 10
            if i >= (self.end_effector_index_list_joint[4] - 2 + 7):
                self.robot_model.lowerPositionLimit[i] = 0
        
        for i in range(len(self.robot_model.upperPositionLimit)):
            if i < 7:
                self.robot_model.upperPositionLimit[i] = 10
            if i >= (self.end_effector_index_list_joint[4] - 2 + 7):
                self.robot_model.upperPositionLimit[i] = 0
            
        lower_pos_lim = self.robot_model.lowerPositionLimit[np.newaxis].T
        upper_pos_lim = self.robot_model.upperPositionLimit[np.newaxis].T
        K_lim = np.identity(self.n_configuration_dimensions)*0.5
        current_config = self.current_joint_config[np.newaxis].T
        lower_pos_lim = np.dot(K_lim,(lower_pos_lim - current_config))
        upper_pos_lim = np.dot(K_lim,(upper_pos_lim - current_config))/self.sampling_time
        lower_pos_lim = lower_pos_lim/self.sampling_time
        lower_pos_lim = np.delete(lower_pos_lim, 0, 0)
        upper_pos_lim = np.delete(upper_pos_lim, 0, 0)
        
        return lower_pos_lim, upper_pos_lim



    def footConstraint(self):
        C = pin.getFrameJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_frame[0], pin.ReferenceFrame.WORLD)#[:3]
        fill = np.zeros((3, self.n_velocity_dimensions))
        #C = np.concatenate((C, fill), axis=0)
        #C[0] = 0
        #C[1] = 0
        #C[2] = 0
        C[3] = 0
        C[4] = 0
        C[5] = 0
        
        for i in range(len(self.end_effector_index_list_frame)-2):
            Jtmp = pin.getFrameJacobian(self.robot_model, self.robot_data, self.end_effector_index_list_frame[i+1], pin.ReferenceFrame.WORLD)#[:3]
            #Jtmp[0] = 0
            #Jtmp[1] = 0
            #Jtmp[2] = 0
            Jtmp[3] = 0
            Jtmp[4] = 0
            Jtmp[5] = 0
            #Jtmp = np.concatenate((Jtmp, fill), axis=0)
            C = np.concatenate((C, Jtmp), axis=0)
        C = np.concatenate((C, np.zeros((6, self.n_velocity_dimensions))), axis=0)
        Clb = np.zeros(C.shape[0]).reshape((C.shape[0],))
        Cub = np.zeros(C.shape[0]).reshape((C.shape[0],))
        
        return C, Clb, Cub


    def CoMConstraint(self): # assumes feet are static
        C= pin.jacobianCenterOfMass(self.robot_model, self.robot_data, self.current_joint_config)[:2]
        C = np.concatenate((C, np.zeros((1, self.n_velocity_dimensions))), axis=0)
        CoM_pos = self.robot_data.com[0]
        FL_pos = self.EE_frame_pos[1] # FL (+x,+y) in local coordinates
        RR_pos = self.EE_frame_pos[2] # RR (-x,-y) in local coordinates
        Clb = ((RR_pos - CoM_pos) / self.dt).reshape(C.shape[0]) * 0.5
        Cub = ((FL_pos - CoM_pos) / self.dt).reshape(C.shape[0]) * 0.5
        return C, Clb, Cub   


    def findConstraints(self):
        C, Clb, Cub = self.footConstraint()
        D, Dlb, Dub = self.CoMConstraint()

        C = np.concatenate((C, D), axis=0)
        Clb = np.concatenate((Clb, Dlb), axis=0).reshape((C.shape[0],))
        Cub = np.concatenate((Cub, Dub), axis=0).reshape((C.shape[0],))

        return C.T, Clb, Cub


    def qpCartesianA(self):
        # this list will contain the jacobians from active tasks
        A_list = []

        # find the jacobains for the active tasks
        if self.taskActiveEE == True:
            self.endEffectorA()
            A_list.append(self.end_effector_A)

        if self.taskActiveTrunk == True:
            self.trunkA()
            A_list.append(self.trunk_A)

        # combine all As to find A
        for i in range(len(A_list)):
            if i == 0:
                A = A_list[i]
            else:
                A = np.concatenate((A, A_list[i]), axis=0)
        
        return A
    

    def EndEffectorB(self, target_cartesian_pos):
        target_list = [0, 0, 0, 0, 0]
        if np.sum(target_cartesian_pos) == 0:
            self.end_effector_B = np.zeros(((self.n_of_EE*6),1))
        else:
            for i in range(len(self.end_effector_index_list_frame)):
                if np.sum(target_cartesian_pos[i]) == 0:
                    target_list[i] = np.zeros((6,1))
                else:
                    if i == 4 and self.initialised == True:
                        self.print_ = True
                    else:
                        self.print_ = False
                    if self.initialised == True:
                        target_list[i] = self.calcTargetVelEE3(target_cartesian_pos[i], self.default_EE_ori_list[i], i, self.EE_gains[i])
                        target_list[i] = target_list[i] * self.cart_task_weight_EE_list[i]
                    else:
                        target_list[i] = self.calcTargetVelEE(target_cartesian_pos[i], self.default_EE_ori_list[i], i, self.EE_gains[i])
                        target_list[i] = target_list[i] * self.cart_task_weight_EE_list[i]
                        
                    self.print_ = False

                    
            self.end_effector_B = target_list[0]
            for i in range(len(target_list)-1):
                self.end_effector_B = np.concatenate((self.end_effector_B,target_list[i+1]), axis=0)


    def TrunkB(self, target_cartesian_pos):
        if self.initialised == True:
            self.trunk_B = self.calcTargetVelTrunk2(target_cartesian_pos, self.default_trunk_ori, self.trunk_frame_index, self.trunk_gain)
            self.trunk_B = self.trunk_B * self.cart_task_weight_Trunk
        else:
            self.trunk_B = self.calcTargetVelTrunk(target_cartesian_pos, self.default_trunk_ori, self.trunk_frame_index, self.trunk_gain)
            self.trunk_B = self.trunk_B * self.cart_task_weight_Trunk
        

        
    def calcTargetVelTrunk(self, target_pos, target_rot, frame_id, gain):
        # find the current cartisian position of the end effector
        fk_pos = np.array([self.robot_data.oMf[frame_id].translation]).T
        fk_ori = np.array([self.robot_data.oMf[frame_id].rotation]).T
        fk_ori = self.Rot2Euler(fk_ori)
        fk_cart = np.concatenate((fk_pos, fk_ori), axis=0)

        # find desired cartesian position
        des_cart = np.concatenate((target_pos, target_rot), axis=0)

        # calculate desired velocity
        pos_vel = (des_cart[:3] - fk_cart[:3]) / self.dt
        ori_vel = np.array([[0,0,0]]).T
        des_vel = np.concatenate((pos_vel,ori_vel), axis=0)

        
        # calculate target end effector velocity
        target_vel = des_vel + np.dot(gain, (des_cart - fk_cart))

        return target_vel


    def calcTargetVelTrunk2(self, target_pos, target_rot, frame_id, gain):
        # find the separate position and orientation gain
        gain_pos = gain[0:3, 0:3]
        gain_ori = gain[3:, 3:]

        """ Find position velocity"""
        # calc target position velocity
        ref_trunk_vel = (target_pos - self.prev_trunk_ref.reshape(3,1))/self.dt

        fk_trunk_pos = np.copy(self.robot_data.oMf[frame_id].translation)
        pos_vel = ref_trunk_vel + (np.dot(gain_pos, (target_pos - fk_trunk_pos.reshape(3,1))).reshape(3,1)/self.dt)

        #ori_vel = np.array([0,0,0]).reshape(3,1)

        """ Find angular velocity"""
        # find the quaternion of the trunk frame through forward kinematics
        fk_trunk_rot = R.from_matrix(np.copy(self.robot_data.oMf[frame_id].rotation))
        fk_trunk_quat = fk_trunk_rot.as_quat()

        # find the referance trunk quaternion
        ref_trunk_rotation = R.from_euler('xyz', target_rot.reshape(3,))
        ref_trunk_rot_matrix = ref_trunk_rotation.as_matrix()
        ref_trunk_quat = ref_trunk_rotation.as_quat()

        # find the error between the ref quaternion and frame quaternion
        quat_error = np.zeros((4,))
        quat_error[0] = (fk_trunk_quat[3]*ref_trunk_quat[0]) - (fk_trunk_quat[0]*ref_trunk_quat[3]) + (fk_trunk_quat[1]*ref_trunk_quat[2]) - (fk_trunk_quat[2]*ref_trunk_quat[1])
        quat_error[1] = (fk_trunk_quat[3]*ref_trunk_quat[1]) - (fk_trunk_quat[1]*ref_trunk_quat[3]) - (fk_trunk_quat[0]*ref_trunk_quat[2]) + (fk_trunk_quat[2]*ref_trunk_quat[0])
        quat_error[2] = (fk_trunk_quat[3]*ref_trunk_quat[2]) - (fk_trunk_quat[3]*ref_trunk_quat[2]) + (fk_trunk_quat[0]*ref_trunk_quat[1]) - (fk_trunk_quat[1]*ref_trunk_quat[0])
        quat_error[3] = (fk_trunk_quat[3]*ref_trunk_quat[3]) + (fk_trunk_quat[0]*ref_trunk_quat[0]) + (fk_trunk_quat[1]*ref_trunk_quat[1]) + (fk_trunk_quat[2]*ref_trunk_quat[2])

        # find angular velocity
        w = np.zeros((3,))
        for ii in range(3):
            w[ii] = -gain_ori[ii,ii] * quat_error[ii]

        skew = np.dot(((ref_trunk_rot_matrix - self.old_ref_trunk_rot_matrix)/self.dt), ref_trunk_rot_matrix)
        trunk_to_CoM_ori_ref = np.zeros((3,))
        trunk_to_CoM_ori_ref[0] = skew[2,1]
        trunk_to_CoM_ori_ref[1] = skew[0,2]
        trunk_to_CoM_ori_ref[2] = skew[1,0]

        ori_vel = (trunk_to_CoM_ori_ref + w).reshape(3,1)

        #ori_vel = w.reshape(3,1)

        # store previouse values
        self.prev_trunk_ref = target_pos
        self.old_ref_trunk_rot_matrix = ref_trunk_rot_matrix

        target_vel = np.concatenate((pos_vel, ori_vel), axis=0)


        return target_vel
        


    def calcTargetVelEE(self, target_pos, target_rot, i, gain):
        # find the separate position and orientation gain
        gain_pos = gain[0:3, 0:3]
        gain_ori = gain[3:, 3:]
        
        # find the current cartisian position of the end effector
        fk_pos = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[i]].translation]).T
        fk_ori = np.array([self.robot_data.oMf[self.end_effector_index_list_frame[i]].rotation]).T
        fk_ori = self.Rot2Euler(fk_ori)
        fk_cart = np.concatenate((fk_pos, fk_ori), axis=0)

        # find desired cartesian position
        des_cart = np.concatenate((target_pos, target_rot), axis=0)

        # calculate desired velocity
        pos_vel = (des_cart[:3] - fk_cart[:3]) / self.dt
        ori_vel = np.array([[0,0,0]]).T
        des_vel = np.concatenate((pos_vel,ori_vel), axis=0)

        
        # calculate target end effector velocity
        target_vel = des_vel + np.dot(gain, (des_cart - fk_cart))

        return target_vel


    def calcTargetVelEE3(self, target_pos, target_rot, i, gain):
        # find the separate position and orientation gain
        gain_pos = gain[0:3, 0:3]
        gain_ori = gain[3:, 3:]
        
        #hip or waist rotation matrix and position
        ref_hip_waist_rot = np.copy(self.robot_data.oMf[self.trunk_frame_index].rotation)
        ref_hip_waist_pos = np.copy(self.robot_data.oMf[self.hip_waist_joint_index_list_frame[i]].translation)

        """ Find postition velocity"""
        # calc target position velocity
        ref_EE_vel = (target_pos - self.prev_EE_pos[i].reshape(3,1))/self.dt

        fk_EE_pos = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[i]].translation)
        #pos_vel = (np.dot(ref_hip_waist_rot.T, (ref_EE_vel - self.hip_waist_vel_list[i].reshape(3,1))) + np.dot(gain_pos, (np.dot(ref_hip_waist_rot.T, target_pos) - fk_EE_pos.reshape(3,1)))).reshape(3,1)
        pos_vel = (np.dot(ref_hip_waist_rot.T, (ref_EE_vel)) + (np.dot(gain_pos, (np.dot(ref_hip_waist_rot.T, target_pos) - np.dot(ref_hip_waist_rot.T, fk_EE_pos.reshape(3,1))))).reshape(3,1)/self.dt)

        #ori_vel = np.array([0,0,0]).reshape(3,1)

        """ Find angular velocity"""
        # find the quaternion of the EE frame through forward kinematics
        fk_EE_rot = R.from_matrix(np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[i]].rotation))
        fk_EE_quat = fk_EE_rot.as_quat()
        
        # find the quaternion of the EE reference
        ref_EE_rotation = R.from_euler('xyz', target_rot.reshape(3,))
        ref_EE_rot = ref_EE_rotation.as_matrix()
        ref_h_R = np.dot(ref_hip_waist_rot.T, ref_EE_rot)
        ref_h_R = ref_EE_rot
        ref_EE_rot_d = R.from_matrix(ref_h_R)
        ref_EE_quat = ref_EE_rot_d.as_quat()

        # find the error between the ref quaternion and frame quaternion
        quat_error = np.zeros((4,))
        quat_error[0] = (fk_EE_quat[3]*ref_EE_quat[0]) - (fk_EE_quat[0]*ref_EE_quat[3]) + (fk_EE_quat[1]*ref_EE_quat[2]) - (fk_EE_quat[2]*ref_EE_quat[1])
        quat_error[1] = (fk_EE_quat[3]*ref_EE_quat[1]) - (fk_EE_quat[1]*ref_EE_quat[3]) - (fk_EE_quat[0]*ref_EE_quat[2]) + (fk_EE_quat[2]*ref_EE_quat[0])
        quat_error[2] = (fk_EE_quat[3]*ref_EE_quat[2]) - (ref_EE_quat[3]*fk_EE_quat[2]) + (fk_EE_quat[0]*ref_EE_quat[1]) - (fk_EE_quat[1]*ref_EE_quat[0])
        quat_error[3] = (fk_EE_quat[3]*ref_EE_quat[3]) + (fk_EE_quat[0]*ref_EE_quat[0]) + (fk_EE_quat[1]*ref_EE_quat[1]) + (fk_EE_quat[2]*ref_EE_quat[2])

        # find angular velocity
        w = np.zeros((3,))
        for ii in range(3):
            w[ii] = -gain_ori[ii,ii] * quat_error[ii]

        # find angular velocity of EE to CoM
        w_ori_to_CoM_ref = np.zeros((3,))
        EE_CoM_Rot = np.dot(ref_hip_waist_rot.T, ref_EE_rot)
        EE_CoM_Rot = ref_EE_rot
        skew = np.dot(((EE_CoM_Rot - self.prev_EE_CoM_rot[i])/self.dt), EE_CoM_Rot.T)
        w_ori_to_CoM_ref[0] = skew[2,1]
        w_ori_to_CoM_ref[1] = skew[0,2]
        w_ori_to_CoM_ref[2] = skew[1,0]

        # find overall anglular velocity
        ori_vel = (w_ori_to_CoM_ref + w).reshape(3,1)
        
        # store previouse values
        self.prev_EE_pos[i] = target_pos
        self.prev_EE_CoM_rot[i] = EE_CoM_Rot

        target_vel = np.concatenate((pos_vel, ori_vel), axis=0)

        return target_vel
        

    def qpCartesianB(self, target_cartesian_pos_EE, target_cartesian_pos_trunk):
        # this list will contain the targets from active tasks
        target_list = []

        # for the active tasks find their targets
        if self.taskActiveEE == True:
            self.EndEffectorB(target_cartesian_pos_EE)
            target_list.append(self.end_effector_B)

        if self.taskActiveTrunk == True:
            self.TrunkB(target_cartesian_pos_trunk)
            target_list.append(self.trunk_B)

        # combine all targets to find b
        for i in range(len(target_list)):
            if i == 0:
                b = target_list[i]
            else:
                b = np.concatenate((b, target_list[i]), axis=0)

        return b


    def qpJointA(self):

        U = np.identity(self.n_velocity_dimensions)
        a = np.ones(self.n_velocity_dimensions)*(1/self.n_velocity_dimensions)
        A = U * a
        A = A * self.joint_task_weight

        return A


    def qpJointb(self):

        # Tikhonov Regularization (default)
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
                J = pin.getJointJacobian(self.robot_model, self.robot_data, joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                f1 = math.sqrt(np.linalg.det(np.dot(J, J.T)))
                q[i] = q[i] - (deltaq * 2)
                self.updateState(q, feedback=False)
                J = pin.getJointJacobian(self.robot_model, self.robot_data, joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                f2 = math.sqrt(np.linalg.det(np.dot(J, J.T)))
                u.append(0.5*(f1-f2)/deltaq)

            u = np.array(u).reshape((self.n_velocity_dimensions, 1))
            self.updateState(self.current_joint_config, feedback=False)

        a = np.ones((self.n_velocity_dimensions, 1))*(1/self.n_velocity_dimensions)

        b = a*u

        b = b * self.joint_task_weight

        return b


    def qpA(self):

        A = self.qpCartesianA()

        if self.taskActiveJoint == True or self.taskActiveJoint == "PREV" or self.taskActiveJoint == "MANI":
            A = np.concatenate((A, self.jointTaskA), axis=0)

        return A


    def qpb(self, target_cartesian_pos_EE, target_cartesian_pos_trunk):
        
        b = self.qpCartesianB(target_cartesian_pos_EE, target_cartesian_pos_trunk)
        
        if self.taskActiveJoint == True or self.taskActiveJoint == "PREV" or self.taskActiveJoint == "MANI":

            b = np.concatenate((b, self.qpJointb()), axis=0)
            
        return b    


    def trunkWorldPos(self):
        WRB = np.copy(self.robot_data.oMf[self.trunk_frame_index].rotation)

        trunk_pos = np.copy(self.robot_data.oMf[self.trunk_frame_index].translation)
        
        FR_WPA = self.FR_target_cartesian_pos
        FR_BPA = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[0]].translation).reshape(3,1) - trunk_pos.reshape(3,1)
        FR_trunk_offset = (FR_WPA - np.dot(WRB, FR_BPA)).reshape(3,)
        
        FL_WPA = self.FL_target_cartesian_pos
        FL_BPA = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[1]].translation).reshape(3,1) - trunk_pos.reshape(3,1)
        FL_trunk_offset = (FL_WPA - np.dot(WRB, FL_BPA)).reshape(3,)

        RR_WPA = self.RR_target_cartesian_pos
        RR_BPA = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[2]].translation).reshape(3,1) - trunk_pos.reshape(3,1)
        RR_trunk_offset = (RR_WPA - np.dot(WRB, RR_BPA)).reshape(3,)

        RL_WPA = self.RL_target_cartesian_pos
        RL_BPA = np.copy(self.robot_data.oMf[self.end_effector_index_list_frame[3]].translation).reshape(3,1) - trunk_pos.reshape(3,1)
        RL_trunk_offset = (RL_WPA - np.dot(WRB, RL_BPA)).reshape(3,)

        trunk_offset = (FR_trunk_offset + FL_trunk_offset + RR_trunk_offset + RL_trunk_offset)/4

        WPA = (self.FR_target_cartesian_pos + self.FL_target_cartesian_pos + self.RR_target_cartesian_pos + self.RL_target_cartesian_pos)/4

        BPA = ((FR_BPA + FL_BPA + RR_BPA + RL_BPA)/4)

        trunk_pos = (WPA - np.dot(WRB, BPA)).reshape(3,)


        return trunk_pos


    def runWBC(self, base_config, target_cartesian_pos_EE=None, target_cartesian_pos_trunk=None):
        # store cartesian pos targets
        self.FR_target_cartesian_pos = target_cartesian_pos_EE[0]
        self.FL_target_cartesian_pos = target_cartesian_pos_EE[1]
        self.RR_target_cartesian_pos = target_cartesian_pos_EE[2]
        self.RL_target_cartesian_pos = target_cartesian_pos_EE[3]

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
        b = self.qpb(target_cartesian_pos_EE, target_cartesian_pos_trunk).reshape((A.shape[0],))

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

        self.jointVelocitiestoConfig(q_vel, True)

        # update robot model with new joint and base configuration
        #self.updateState(joint_config, base_config, running=True)

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
    
