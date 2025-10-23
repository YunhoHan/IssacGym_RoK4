# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os, time
import json
import cv2


from isaacgym import gymtorch
from isaacgym import gymapi
from .base.vec_task import VecTask

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Joy


import torch
from typing import Tuple, Dict
import random
from isaacgymenvs.utils.torch_jit_utils import quat_from_euler_xyz, quat_rotate,my_quat_rotate,quat_conjugate, to_torch, get_axis_params, torch_rand_float, normalize, quat_apply, quat_rotate_inverse,get_euler_xyz,matrix_to_quaternion,quat_mul
from isaacgymenvs.tasks.base.vec_task import VecTask

class RoK3(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.height_samples = None
        self.custom_origins = False
        self.init_done = False
        self.need_reset = False
        
        # self.num_obs = self.cfg["env"]["numObservations"]
        self.num_legs = 2
    
        self.freeze_cnt = 0
        self.freeze_flag = False
        self.freeze_steps = 50

        self.min_swing_time = 0.15
        self.cycle_time = 1.

        self.ref_torques = 0.
        self.plot_cnt = 1

        self.cam_mode = 0
        self.smoothed_cam_pos = None
        self.smoothed_cam_target = None
        self.smoothing_alpha = 0.1

        self.test_mode = self.cfg["env"]["test"]

        if self.test_mode:
            self.observe_envs = 0
            self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
            self.cam_change_flag = True
            self.cam_change_cnt = 0
            print('\033[103m' + '\033[33m' + '_________________________________________________________' + '\033[0m')
            print('\033[103m' + '\033[33m' + '________________________' + '\033[30m' + 'Test Mode' + '\033[33m' + '\033[103m'+ '________________________' + '\033[0m')
            print('\033[103m' + '\033[33m' +'_________________________________________________________' + '\033[0m')
        else:
            self.observe_envs = 0

        rospy.init_node('plot_juggler_node')

        # obs_scales
        self.obs_scales = {}
        for key, value in self.cfg["env"]["learn"]["obs_scales"].items():  # rewards 섹션 순회
            self.obs_scales[key] = float(value) 

        # rew_scales
        self.rew_scales = {}
        self.reward_container ={}
        for key, value in self.cfg["env"]["learn"]["reward"].items():  # rewards 섹션 순회
            self.rew_scales[key] = float(value) 

        # action_scale
        self.action_scale       = self.cfg["env"]["control"]["actionScale"]
        self.soft_dof_pos_limit = self.cfg["env"]["learn"]["soft_dof_pos_limit"]

        # randomization
        self.randomize              = self.cfg["task"]["randomize"]
        self.randomization_params   = self.cfg["task"]["randomization_params"]
        self.delay_proprioceptive   = self.cfg["task"]["delay_proprioceptive"]

        #command ranges
        self.command_x_range    = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range    = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range  = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # envs ranges
        self.stand_env_range                = [0, self.cfg["env"]["EnvsNumRanges"]["stand_env_range"] - 1]
        self.only_plus_x_envs_range         = [self.stand_env_range[1] + 1, self.stand_env_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_plus_x_envs_range"]]
        self.only_minus_x_envs_range        = [self.only_plus_x_envs_range[1] + 1, self.only_plus_x_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_minus_x_envs_range"]]
        self.only_plus_y_envs_range         = [self.only_minus_x_envs_range[1] + 1, self.only_minus_x_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_plus_y_envs_range"]]
        self.only_minus_y_envs_range        = [self.only_plus_y_envs_range[1] + 1, self.only_plus_y_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_minus_y_envs_range"]]
        self.only_plus_yaw_envs_range       = [self.only_minus_y_envs_range[1] + 1, self.only_minus_y_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_plus_yaw_envs_range"]]
        self.only_minus_yaw_envs_range      = [self.only_plus_yaw_envs_range[1] + 1, self.only_plus_yaw_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["only_minus_yaw_envs_range"]]
        self.plus_x_plus_yaw_envs_range     = [self.only_minus_yaw_envs_range[1] + 1, self.only_minus_yaw_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["plus_x_plus_yaw_envs_range"]]
        self.plus_x_minus_yaw_envs_range    = [self.plus_x_plus_yaw_envs_range[1] + 1, self.plus_x_plus_yaw_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["plus_x_minus_yaw_envs_range"]]
        self.minus_x_plus_yaw_envs_range    = [self.plus_x_minus_yaw_envs_range[1] + 1, self.plus_x_minus_yaw_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["minus_x_plus_yaw_envs_range"]]
        self.minus_x_minus_yaw_envs_range   = [self.minus_x_plus_yaw_envs_range[1] + 1, self.minus_x_plus_yaw_envs_range[1] + self.cfg["env"]["EnvsNumRanges"]["minus_x_minus_yaw_envs_range"]]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)


        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.policy_dt = (self.decimation) * self.cfg["sim"]["dt"]
        self.sim_dt = self.cfg["sim"]["dt"]

        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.policy_dt + 0.5)

        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.policy_dt + 0.5)
        self.freeze_interval = int(self.cfg["env"]["learn"]["freezecmdInterval_s"] / self.policy_dt + 0.5)

        self.Kp = self.cfg["env"]["control"]["Kp"]
        self.Kd = self.cfg["env"]["control"]["Kd"]

        self.Kp = torch.tensor(self.Kp, dtype=torch.float32, device=self.device)
        self.Kd = torch.tensor(self.Kd, dtype=torch.float32, device=self.device)
        print('\033[94m',"Kp : ", self.Kp , '\033[0m')
        print('\033[94m',"Kd : ", self.Kd , '\033[0m')
        print('\033[94m',"Kp : ", self.Kp.size() , '\033[0m')
        print('\033[94m',"Kd : ", self.Kd.size() , '\033[0m')
        # print('\033[92m' + "Kp size: " + self.Kp.size() + '\033[0m')
        # print('\033[92m' + "Kd size: " + self.Kd.size() + '\033[0m')

        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.policy_dt

        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state    = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor    = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces  = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_tensor   = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # fsdata              = self.gym.acquire_force_sensor_tensor(self.sim)
        # camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs, self.camera_handles, gymapi.IMAGE_COLOR)


        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)

        # self.fsdata = gymtorch.wrap_tensor(fsdata)
        # self.foot_force  = self.fsdata.view(self.num_envs, 2, 6)[..., :3]
        # self.foot_torque = self.fsdata.view(self.num_envs, 2, 6)[...,3:7]

        # create some wrapper tensors for different slices
        self.root_states    = gymtorch.wrap_tensor(actor_root_state)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        
        # self.camera_tensor = gymtorch.wrap_tensor(camera_tensor)
        
        # dof state
        self.dof_state  = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos    = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel    = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        # rigid body state
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor)
        self.rigid_body_pos   = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,:3]
        self.rigid_body_rot   = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,3:7]
        self.rigid_body_vel   = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,7:10]


        # initialize some data used later on
        self.common_step_counter = 0

        self.extras = {}
        
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_x          = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_y          = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_yaw        = self.commands.view(self.num_envs, 3)[..., 2] 


        # Base factors
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.base_pos = torch.zeros(self.num_envs,3, device=self.device, dtype=torch.float)
        self.base_vel = torch.zeros(self.num_envs,3, device=self.device, dtype=torch.float)
        self.last_base_vel = torch.zeros(self.num_envs,3, device=self.device, dtype=torch.float)
        self.robot_CoM = torch.zeros(self.num_envs,3, device=self.device, dtype=torch.float)


        self.body_CoM = torch.zeros(
            self.num_envs,self.num_bodies,3,
            dtype= torch.float32,
            device= self.device,
        )

        self.total_mass = torch.zeros(
            self.num_envs,
            dtype= torch.float32,
            device= self.device,
        )

        self.body_mass = torch.zeros(
            self.num_envs,self.num_bodies,
            dtype= torch.float32,
            device= self.device,
        )

        self.body_mass_noise = torch.zeros(
            self.num_envs,self.num_bodies,
            dtype= torch.float32,
            device= self.device,
        )

        self.foot_swing_start_time = torch.zeros(self.num_envs, 2, dtype=torch.float,device=self.device)
        self.foot_swing_state = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)  # 각 발의 스윙 상태
        self.projected_gravity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # Dof factors
        ## ㄴActions
        self.torques = torch.zeros(self.num_envs, self.num_actions-4, dtype=torch.float, device=self.device, requires_grad=False)
        self.targets = torch.zeros_like(self.dof_pos)
        self.actions = torch.zeros(self.num_envs, self.num_actions-4, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_actions = torch.zeros(self.num_envs,4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_clock_actions = torch.zeros(self.num_envs,4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions-4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions2 = torch.zeros(self.num_envs, self.num_actions-4, dtype=torch.float, device=self.device, requires_grad=False)
        
        ## ㄴDof
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_vel2 = torch.zeros_like(self.dof_vel)
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.last_dof_acc = torch.zeros_like(self.dof_vel)

        # Foot factors
        ## ㄴCycle
        self.sin_cycle = torch.zeros(self.num_envs,self.num_legs, dtype=torch.float, device=self.device, requires_grad=False)
        self.cos_cycle = torch.zeros(self.num_envs,self.num_legs, dtype=torch.float, device=self.device, requires_grad=False)
        self.cycle_t = torch.zeros(self.num_envs,2, device=self.device, dtype=torch.float)
        self.cycle_L_x = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.cycle_R_x = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.phi = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float,device=self.device)

        ## ㄴFoot State
        self.ref_foot_height = torch.zeros(self.num_envs,self.num_legs, device=self.device, dtype=torch.float)
        self.last_foot_contacts = torch.zeros(self.num_envs,self.num_legs, device=self.device, dtype=torch.bool)
        self.foot_pos = torch.zeros(self.num_envs,self.num_legs, device=self.device, dtype=torch.float)
        self.foot_air_time = torch.zeros(self.num_envs, self.num_legs, device=self.device, dtype=torch.float)

        # Etc
        self.no_commands = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.height_points = self.init_height_points()
        self.leg_height_points = self.init_leg_height_points()
        self.measured_heights = None
        self.measured_legs_heights = None

        # Reset
        self.body_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dof_limit_lower = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dof_limit_upper = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dof_vel_over = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        for i in range(self.num_actions-4):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(), "ang_vel_xy": torch_zeros(),
                             "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(), "base_height": torch_zeros(),
                             "air_time": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros(), "action_rate": torch_zeros(), "hip": torch_zeros()}

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done = True


    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        terrain_type = self.cfg["env"]["terrain"]["terrainType"] 
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = True
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        if self.randomize:
            print('\033[102m' + '\033[32m' + '_________________________________________________________' + '\033[0m')
            print('\033[102m' + '\033[32m' + '_____________________' + '\033[30m' + 'Randomize True' + '\033[32m' + '\033[102m'+ '______________________' + '\033[0m')
            print('\033[102m' + '\033[32m' +'_________________________________________________________' + '\033[0m')
            self.apply_randomizations(self.randomization_params)
        else:
            print('\033[101m' + '\033[31m' + '_________________________________________________________' + '\033[0m')
            print('\033[101m' + '\033[31m' + '______________________' + '\033[30m' + 'Randomize False' + '\033[31m' + '\033[101m'+ '______________________' + '\033[0m')
            print('\033[101m' + '\033[31m' +'_________________________________________________________' + '\033[0m')
    
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise      = self.cfg["env"]["learn"]["noise"]["addNoise"]
        noise_level         = self.cfg["env"]["learn"]["noise"]["noiseLevel"]
        noise_vec[0:3]      = self.cfg["env"]["learn"]["noise"]["angularVelocityNoise"] * noise_level * self.obs_scales["angularVelocityScale"] # base ang vel
        noise_vec[3:6]      = self.cfg["env"]["learn"]["noise"]["gravityNoise"] * noise_level # projected gravity
        noise_vec[6:9]      = 0. # commands
        noise_vec[9:22]     = self.cfg["env"]["learn"]["noise"]["dofPositionNoise"] * noise_level * self.obs_scales["dofPositionScale"] # dof pos
        noise_vec[22:35]    = self.cfg["env"]["learn"]["noise"]["dofVelocityNoise"] * noise_level * self.obs_scales["dofVelocityScale"] # dof vel
        noise_vec[35:48]    = 0. # previous actions
        noise_vec[48:55]    = 0. # sin/cos cycle
        noise_vec[55:57]    = self.cfg["env"]["learn"]["noise"]["ftsensorNoise"] * noise_level * self.obs_scales["ftSensorScale"] # dof vel

        return noise_vec
    

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size 
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        
        asset_file = self.cfg["env"]["urdfAsset"]["file"]
        if self.test_mode:
            asset_file = self.cfg["env"]["urdfAsset"]["test_file"]
        
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = False
        asset_options.replace_cylinder_with_capsule = False
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.use_mesh_materials = False
        asset_options.disable_gravity = False

        rok3_asset      = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof    = self.gym.get_asset_dof_count(rok3_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(rok3_asset)
        
        # prepare friction randomization
        self.rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(rok3_asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        print('\033[93m' + "friction_range : " + '\033[0m', friction_range)
        num_buckets = 500
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)
        
        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.body_names = self.gym.get_asset_rigid_body_names(rok3_asset)
        self.dof_names  = self.gym.get_asset_dof_names(rok3_asset)
        foot_name       = self.cfg["env"]["urdfAsset"]["footName"]
        calf_name       = self.cfg["env"]["urdfAsset"]["calfName"]
        self.depth_camera_attach_body_name = self.cfg["env"]["urdfAsset"]["camera_attach_link"]

        self.cam_link_index = self.body_names.index(self.depth_camera_attach_body_name)

        feet_names = [s for s in self.body_names if foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        calf_names = [s for s in self.body_names if calf_name in s]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(rok3_asset)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device,requires_grad=False)
        self.reset_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,requires_grad=False)

        for i in range(len(dof_props)):
            self.dof_pos_limits[i, 0] = dof_props["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props["upper"][i].item()
            self.reset_dof_pos_limits[i, 0] = self.dof_pos_limits[i, 0]
            self.reset_dof_pos_limits[i, 1] = self.dof_pos_limits[i, 1]
            self.dof_vel_limits[i] = dof_props["velocity"][i].item()

            dof_props['stiffness'][i] = 0
            # dof_props['friction'][i] = 0.02
            dof_props['armature'][i] = 0.01

        dof_props['damping'] = self.cfg["env"]["dof_porperties"]["damping"]
        dof_props['friction'] = self.cfg["env"]["dof_porperties"]["friction"]

        self.joint_torque_limits = torch.tensor(self.cfg["env"]["dof_porperties"]["torque_limits"] , device=self.device) # device 설정 필요
        # print("1 : ",self.joint_torque_limits)
        self.joint_torque_limits *= 0.9
        # print("2 : ",self.joint_torque_limits  )

        dof_property_names = [
            "hasLimits", 
            "lower", 
            "upper", 
            "driveMode", 
            "velocity", 
            "effort", 
            "stiffness", 
            "damping", 
            "friction", 
            "armature"
        ]

        for i, dof_prop in enumerate(dof_props):
            print(f"관절 {i+1}:")
            for j, value in enumerate(dof_prop):
                print(f"  {dof_property_names[j]}: {value}") 

        # # [뎁스 카메라 추가] 카메라 속성 정의 (설정 파일에서 로드)
        # self.camera_props = gymapi.CameraProperties()
        # self.camera_props.width = int(self.cfg["env"]["sensor"]["forward_camera"]["width"])
        # self.camera_props.height = int(self.cfg["env"]["sensor"]["forward_camera"]["height"])
        # self.camera_props.horizontal_fov = self.cfg["env"]["sensor"]["forward_camera"]["horizontal_fov"]
        # self.camera_props.enable_tensors = True # GPU 텐서 사용 (최신 Isaac Gym에서는 자동 설정될 수 있음)
        # self.camera_props.near_plane = self.cfg["env"]["sensor"]["forward_camera"]["near_plane"]
        # self.camera_props.far_plane = self.cfg["env"]["sensor"]["forward_camera"]["far_plane"]
        
        # self.debug_img = self.cfg["env"]["sensor"]["forward_camera"]["debug_img"]
        # # [뎁스 카메라 추가] 각 환경별 카메라 핸들 및 뎁스 이미지 버퍼
        # self.camera_handles = [0] * self.num_envs # 리스트 크기 미리 할당
        # self.depth_image_buf = torch.zeros(
        #     (self.num_envs, self.camera_props.height, self.camera_props.width),
        #     device=self.device, dtype=torch.float32 # 뎁스 이미지는 float32
        # )


        # # [뎁스 카메라 추가] 카메라 부착 정보
        # self.depth_camera_local_transform = gymapi.Transform()
        # cam_pos_cfg = self.cfg["env"]["sensor"]["forward_camera"]["position"]
        # cam_rot_cfg = self.cfg["env"]["sensor"]["forward_camera"]["rotation"] # Euler XYZ (degrees)
        # self.depth_camera_local_transform.p = gymapi.Vec3(cam_pos_cfg[0], cam_pos_cfg[1], cam_pos_cfg[2])
        # # Euler 각도를 쿼터니언으로 변환 (도 -> 라디안 변환 포함)
        # q_cam = quat_from_euler_xyz(
        #     torch.deg2rad(torch.tensor(cam_rot_cfg[0], dtype=torch.float32, device=self.device)),
        #     torch.deg2rad(torch.tensor(cam_rot_cfg[1], dtype=torch.float32, device=self.device)),
        #     torch.deg2rad(torch.tensor(cam_rot_cfg[2], dtype=torch.float32, device=self.device))
        # )
        # self.depth_camera_local_transform.r = gymapi.Quat(q_cam[0].item(), q_cam[1].item(), q_cam[2].item(), q_cam[3].item())


        
        # # [뎁스 카메라 추가] 관찰 공간 크기 업데이트 (필요시 VecTask.__init__ 전)
        # # 예: self.cfg["env"]["numObservations"] += self.camera_props.width * self.camera_props.height
        # # 이 값은 VecTask의 observation_space, obs_buf 크기 등에 영향을 줍니다.
        

        # L_Foot_idx = self.gym.find_asset_rigid_body_index(rok3_asset, "L_Foot")
        # R_Foot_idx = self.gym.find_asset_rigid_body_index(rok3_asset, "R_Foot")

        # sensor_pose1 = gymapi.Transform(gymapi.Vec3(0., 0.0, 0.009))
        # sensor_pose2 = gymapi.Transform(gymapi.Vec3(0., 0.0, 0.009))

        # sensor_props = gymapi.ForceSensorProperties()
        # sensor_props.enable_forward_dynamics_forces = True     
        # sensor_props.enable_constraint_solver_forces = True
        # sensor_props.use_world_frame = False
        # # print("L_Foot_idx : ", L_Foot_idx)

        # # Asset에 센서 추가 (이것은 Asset 정의에 추가하는 것이므로 루프 밖에서 한 번만 수행)
        # self.sensor_idx1 = self.gym.create_asset_force_sensor(rok3_asset, L_Foot_idx, sensor_pose1, sensor_props)
        # self.sensor_idx2 = self.gym.create_asset_force_sensor(rok3_asset, R_Foot_idx, sensor_pose2, sensor_props)

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.
    
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.rok3_handles = []
        self.envs = []
        self.cameras = []
        self.cam_tensors = []

        self.robot_config_buffer = torch.empty(
            self.num_envs,self.num_bodies,4,
            dtype= torch.float32,
            device= self.device,
        )        

        # --- 1단계: 환경 및 액터 생성 ---
        print("액터 생성 시작...")
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            rok3_handle = self.gym.create_actor(env_handle, rok3_asset, start_pose, "RoK3", i, 0, 0)
            
            # # [뎁스 카메라 추가] 카메라 센서 생성 및 부착
            # cam_handle = self.gym.create_camera_sensor(env_handle, self.camera_props)
            # # # 액터 내의 특정 바디 핸들 가져오기
            # body_handle = self.gym.get_actor_rigid_body_handle(env_handle, rok3_handle, self.cam_link_index)


            # self.gym.attach_camera_to_body(
            #         cam_handle,                   # camera_handle
            #         env_handle,                   # env_handle
            #         body_handle,                  # body_handle to attach to
            #         self.depth_camera_local_transform, # local transform of camera
            #         gymapi.FOLLOW_TRANSFORM       # follow mode
            #     )
            # self.camera_handles[i] = cam_handle # 핸들 저장

            # # # cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_COLOR)
            # cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH)
            
            # torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
            # self.cam_tensors.append(torch_cam_tensor)


            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

            self.envs.append(env_handle)
            self.rok3_handles.append(rok3_handle)
            # self.cameras.append(cam_handle)
        
        print("액터 생성 완료.")
       
        # --- 각 액터의 마찰력 속성 설정 ---
        print("액터별 마찰력 설정 시작...")
        for i in range(self.num_envs):
            env_handle = self.envs[i]
            rok3_handle = self.rok3_handles[i]

            # 현재 액터의 rigid shape 속성
            current_props = self.gym.get_actor_rigid_shape_properties(env_handle, rok3_handle)

            # 이 환경(i)에 적용할 마찰력 계산
            target_friction = friction_buckets[i % num_buckets]

            # 이 액터의 *모든* rigid shape에 대해 마찰력 설정
            # (만약 특정 shape만 바꾸려면 인덱스를 사용해야 함: current_props[7].friction = ..., current_props[14].friction = ...)
            for prop in current_props:
                prop.friction = target_friction
            
            actor_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, rok3_handle)
            for j in range(self.num_bodies):
                self.robot_config_buffer[i,j,0] = actor_rigid_body_props[j].com.x
                self.robot_config_buffer[i,j,1] = actor_rigid_body_props[j].com.y
                self.robot_config_buffer[i,j,2] = actor_rigid_body_props[j].com.z
                self.robot_config_buffer[i,j,3] = actor_rigid_body_props[j].mass
            # 변경된 속성을 *이 액터에만* 다시 적용
            self.gym.set_actor_rigid_shape_properties(env_handle, rok3_handle, current_props)
        print("robot_config_buffer[CoM_x,CoM_y,CoM_z,mass] : ", self.robot_config_buffer[0])
        print("액터별 마찰력 설정 완료.")

        # Color
        # if self.test_mode :
        #     for j in range(self.num_bodies):
        #         self.gym.set_rigid_body_color(
        #             env_handle, rok3_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.420, 0.420, 0.420))
            
        #     for i in range(1, 5):  # i가 1부터 시작하도록 수정
        #         i *= 4
        #         self.gym.set_rigid_body_color(
        #             env_handle, rok3_handle, i, gymapi.MESH_VISUAL, gymapi.Vec3(0.996, 0.890, 0.420))

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.rok3_handles[0], feet_names[i])
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.rok3_handles[0], calf_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.rok3_handles[0], "base")
        
    
    def check_termination(self):
        reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.body_contact = torch.any(torch.norm(self.contact_forces[:, [0, 1, 2, 3, 8, 9, 10, 15], :], dim=-1) > 1., dim=1)
        body_contact_reset = self.body_contact
        reset = reset | body_contact_reset

        if not (self.test_mode):
            self.time_out = self.progress_buf >= self.max_episode_length - 1
            time_out_reset = self.time_out
            reset = reset | time_out_reset

        # print("self.progress_buf : ", self.progress_buf)
        # print("self.time_out : ", self.time_out)

        if self.test_mode:
            need_reset_reset = self.need_reset
            reset = reset | need_reset_reset

        self.fall = torch.any(self.rigid_body_pos[:, 0, 2].unsqueeze(1) < self.rigid_body_pos[:, [7, 14], 2], dim=-1)
        reset = reset | self.fall

        self.fall2 = (self.rigid_body_pos[:, 0, 2] < 0.3) | (self.rigid_body_pos[:, 0, 2] > 1.5)
        reset = reset | self.fall2


        # self.dof_limit_lower = torch.any(self.dof_pos <= self.reset_dof_pos_limits[:, 0], dim=-1)
        # dof_limit_lower_reset = self.dof_limit_lower
        # reset = reset | dof_limit_lower_reset

        # self.dof_limit_upper = torch.any(self.dof_pos >= self.reset_dof_pos_limits[:, 1], dim=-1)
        # dof_limit_upper_reset = self.dof_limit_upper
        # reset = reset | dof_limit_upper_reset

        return reset
    
    @torch.no_grad()
    def compute_observations(self):
        # proprioceptive info
        base_quat = self.root_states[:, 3:7]
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13]) * self.obs_scales["angularVelocityScale"]

        self.projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        
        dof_pos_scaled = self.dof_pos * self.obs_scales["dofPositionScale"]
        dof_vel_scaled = self.dof_vel * self.obs_scales["dofVelocityScale"]

        # self.phi = torch.tensor([0., torch.pi], device=self.device, dtype=torch.float)
        phi = self.clock_actions[:,:2] * ~self.no_commands.unsqueeze(1)

        # print("self.phi : ",self.phi.size())
        # print("phi : ",phi1.size())
        # self.sin_cycle = torch.sin(2.*torch.pi*(self.cycle_t.unsqueeze(1)/self.cycle_time) + self.phi)
        # self.cos_cycle = torch.cos(2.*torch.pi*(self.cycle_t.unsqueeze(1)/self.cycle_time) + self.phi)

        # print("self.cycle_t : ", self.cycle_t.size())

        cycle_t_L = self.cycle_t[:,0]
        cycle_t_R = self.cycle_t[:,1]

        self.cycle_L_x = 2.*torch.pi*(cycle_t_L/self.cycle_time) + phi[:,0]
        self.cycle_R_x = 2.*torch.pi*(cycle_t_R/self.cycle_time) + phi[:,1]+ torch.pi


        self.sin_cycle[:,0] = torch.sin(self.cycle_L_x)
        self.cos_cycle[:,0] = torch.cos(self.cycle_L_x)

        self.sin_cycle[:,1] = torch.sin(self.cycle_R_x)
        self.cos_cycle[:,1] = torch.cos(self.cycle_R_x)

        # print("self.cycle_t : ", self.cycle_t[self.observe_envs])
        # print("self.sin_cycle : ",self.sin_cycle[0])
        # print("self.cos_cycle : ",self.cos_cycle[0])
        # print("sin_cycle1 : ",sin_cycle1.size())
        # print("cos_cycle1 : ",cos_cycle1.size())
        # foot_force_z  = self.foot_force[:,:,2] * self.obs_scales["ftSensorScale"]
        # foot_torque_x = self.foot_torque[:,:,0] * self.obs_scales["ftSensorScale"]
        # foot_torque_y = self.foot_torque[:,:,1] * self.obs_scales["ftSensorScale"]

        foot_contact_forces_z = self.contact_forces[:, self.feet_indices, 2]

        self.obs_buf = torch.cat((base_ang_vel,                 # 3   [0:3]
                                             self.projected_gravity,       # 3   [3:6]
                                             
                                             self.commands[:, :3],         # 3   [6:9]
                        
                                             dof_pos_scaled,               # 13  [9:22]                 
                                             dof_vel_scaled,               # 13  [22:35]  
                                             self.actions,                 # 13  [35:48]
                                             self.sin_cycle,               # 2   [48:50]
                                             self.cos_cycle,               # 2   [50:52]
                                             self.clock_actions,            # 3   [52:55]
                                            #  foot_contact_forces_z,        # 2   [53:55]

                                             ), dim=-1)                    # 58
        
                
        # # privileged info
        # self.total_mass      = torch.sum(self.body_mass,dim=-1)
        # body_CoM_noise = (torch.rand_like(self.body_CoM)*0.1) - 0.05
        # for i in range(self.num_bodies):
        #     self.body_CoM[:,i,:] = my_quat_rotate(self.base_quat,self.robot_config_buffer[:,i,:3]) 
        #     self.body_CoM[:,i,:] += self.rigid_body_pos[:,i,:]
        # self.body_CoM += body_CoM_noise

        # self.robot_CoM = torch.sum(self.body_mass.unsqueeze(-1) * self.body_CoM, dim=1) / self.total_mass.unsqueeze(-1)
        # # print("robot_CoM : ", robot_CoM[0])
        
        # distance = self.root_states[: , :2] - self.env_origins[:, :2]
        # base_posi = torch.cat((distance, self.root_states[:, 2:3]), dim=-1)

        # base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        # base_ori = base_quat

        # foot_contact =  torch.any(self.contact_forces[:, self.feet_indices, :],dim=-1) > 1.

        # self.privileged_buf = torch.cat((base_posi,                # 3
        #                                  base_lin_vel,                    # 3
        #                                  self.rigid_body_pos[:,[7,14],0], # 2
        #                                  self.rigid_body_pos[:,[7,14],1], # 2
        #                                  self.rigid_body_pos[:,[7,14],2], # 2
        #                                  ),dim=-1)                        # 9

        # if self.randomize & self.delay_proprioceptive:
        #     self.delayed_proprioceptive()
        #     proprioceptive_buf = self.delayed_proprioceptive_buf + (2 * torch.rand_like(self.delayed_proprioceptive_buf) - 1) * self.noise_scale_vec
        # else:
        # proprioceptive_buf = self.proprioceptive_buf
        
        # self.obs_buf = torch.cat((proprioceptive_buf,             # 57
        #                             ),dim=-1)
        
        # self.states_buf = torch.cat((self.obs_buf, 
        #                              self.privileged_buf), dim=-1)

    # @torch.no_grad()
    # def compute_observations_step(self):
    #     # proprioceptive info
    #     base_quat = self.root_states[:, 3:7]
    #     base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13]) * self.obs_scales["angularVelocityScale"]

    #     self.projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        
    #     dof_pos_scaled = self.dof_pos * self.obs_scales["dofPositionScale"]
    #     dof_vel_scaled = self.dof_vel * self.obs_scales["dofVelocityScale"]

    #     self.phi = torch.tensor([0., torch.pi], device=self.device, dtype=torch.float)
    #     self.sin_cycle = torch.sin(2.*torch.pi*(self.cycle_t.unsqueeze(1)/self.cycle_time) + self.phi)
    #     self.cos_cycle = torch.cos(2.*torch.pi*(self.cycle_t.unsqueeze(1)/self.cycle_time) + self.phi)
        
    #     foot_force_z  = self.foot_force[:,:,2] * self.obs_scales["ftSensorScale"]
    #     foot_torque_x = self.foot_torque[:,:,0] * self.obs_scales["ftSensorScale"]
    #     foot_torque_y = self.foot_torque[:,:,1] * self.obs_scales["ftSensorScale"]

    #     self.proprioceptive_buf = torch.cat((base_ang_vel,                 # 3   [0:3]
    #                                          self.projected_gravity,       # 3   [3:6]
                                             
    #                                          self.commands[:, :3],         # 3   [6:9]

    #                                          dof_pos_scaled,               # 13  [9:22]                 
    #                                          dof_vel_scaled,               # 13  [22:35]  
    #                                          self.actions,                 # 13  [35:48]
    #                                          self.sin_cycle,               # 2   [48:50]
    #                                          self.cos_cycle,               # 2   [50:52]


    #                                          foot_force_z,                 # 2   [52:54]
    #                                          foot_torque_x,                # 2   [54:56]
    #                                          foot_torque_y,                # 2   [56:58]
    #                                          ), dim=-1)                    # 41
                
    #     # privileged info
    #     distance = self.root_states[: , :2] - self.env_origins[:, :2]
    #     base_posi = torch.cat((distance, self.root_states[:, 2:3]), dim=-1)

    #     base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
    #     base_ori = base_quat

    #     foot_contact =  torch.any(self.contact_forces[:, self.feet_indices, :],dim=-1) > 1.


    #     self.privileged_buf = torch.cat((base_posi,
    #                                      base_lin_vel,                    # 3
    #                                      self.rigid_body_pos[:,[7,14],0], # 2
    #                                      self.rigid_body_pos[:,[7,14],1], # 2
    #                                      self.rigid_body_pos[:,[7,14],2], # 2
    #                                      ),dim=-1)                        # 9


            
    #     self.obs_buf = torch.cat((self.proprioceptive_buf,      # 57
    #                                 self.privileged_buf,          # 10  [58:68]
    #                                 ),dim=-1)

    # @torch.no_grad()
    # def delayed_proprioceptive(self):
    #     # latency_range = self.cfg["env"]["learn"]["sensorLatency"]
    #     # latency = torch_rand_float(latency_range[0], latency_range[1], (self.num_envs,1), device=self.device)
    #     # buffer_delayed_frames = (latency / self.sim_dt).long().squeeze(-1) 

    #     # proprioceptive 관찰 벡터 내의 각 컴포넌트 범위 정의
    #     # 사용자님의 설명을 기반으로 인덱스 범위 설정
    #     RANGE_ANG_GRAV = (0, 6)       # base_ang_vel (0:3), projected_gravity (3:6) -> 센서 그룹 1 (예: IMU)
    #     RANGE_COMMANDS = (6, 9)       # commands (6:9) -> 딜레이 0
    #     RANGE_DOF_POS_VEL = (9, 35)   # dof_pos_scaled (9:22), dof_vel_scaled (22:35) -> 센서 그룹 2 (예: 관절 엔코더)
    #     RANGE_ACTIONS = (35, 48)      # actions (35:48) -> 딜레이 0
    #     RANGE_CYCLE = (48, 50)        # sin_cycle (48:50), cos_cycle (50:52) -> 딜레이 0
    #     RANGE_FOOT_SENSORS = (50, 54) # foot_force_z (52:54), foot_torque_x (54:56), foot_torque_y (56:58) -> 센서 그룹 3 (예: 발 접촉/힘 센서)

    #     # 각 관찰 범위와 해당 센서 타입 이름을 매핑
    #     range_to_sensor_type = {
    #     RANGE_ANG_GRAV: "sensor_imu",
    #     RANGE_COMMANDS: "zero_delay", # 특수 키 또는 값으로 딜레이 0 표시
    #     RANGE_DOF_POS_VEL: "sensor_joint_encoders",
    #     RANGE_ACTIONS: "zero_delay",
    #     RANGE_CYCLE: "zero_delay",
    #     RANGE_FOOT_SENSORS: "sensor_foot_contact",
    # }
    #     # 센서 타입별 지연 범위 설정 가져오기 (cfg에서 읽어온다고 가정)
    #     sensor_latencies_cfg = self.cfg["env"]["learn"].get("sensor_latencies", {})

    #     sensor_type_to_delay_frames = {}

    #     # 딜레이 0에 해당하는 지연 스텝 텐서 생성 (인덱스 0)
    #     sensor_type_to_delay_frames["zero_delay"] = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    #     # 딜레이가 필요한 각 센서 타입에 대해
    #     for sensor_type, latency_range in sensor_latencies_cfg.items():
    #         if sensor_type == "zero_delay": # zero_delay는 위에서 처리
    #             continue

    #         # 해당 센서 타입의 지연 범위에서 샘플링
    #         latency = torch_rand_float(latency_range[0], latency_range[1], (self.num_envs, 1), device=self.device)
            
    #         # 지연 시간(초)을 시뮬레이션 스텝 수로 변환
    #         delay_frames = (latency / self.sim_dt).long().squeeze(-1) # (num_envs,) 형태의 정수 텐서
            
    #         # 버퍼의 최대 길이보다 커지지 않도록 클리핑 (선택적이지만 안정성 위해 권장)
    #         max_buffer_index = self.last_proprioceptive_bufs.shape[0] - 1
    #         delay_frames = torch.clip(delay_frames, 0, max_buffer_index)

    #         # 결과 저장
    #         sensor_type_to_delay_frames[sensor_type] = delay_frames


    #     # --- 디버깅을 위한 출력 ---
    #     # 각 센서 타입별 계산된 지연 스텝 (각 환경별 값)
    #     # print("--- Calculated Delay Frames per Sensor Type ---")
    #     # for sensor_type, delay_frames_tensor in sensor_type_to_delay_frames.items():
    #     #     print(f"Sensor Type: {sensor_type}")
    #     #     # 텐서 내용이 길면 잘릴 수 있으므로 전체 출력 설정을 확인
    #     #     print(f"  Delay Index (for each env): {delay_frames_tensor}") # 0이 최신, 양수가 오래된 값
    #     # print("-" * 30)

    #     # # 원본 과거 관찰 버퍼 내용 확인 (예: 0번 환경의 첫 10개 스텝)
    #     # # self.last_proprioceptive_bufs 형태: (buffer_length, num_envs, num_proprioceptive_obs)
    #     # num_steps_to_print = min(self.last_proprioceptive_bufs.shape[0], 10) # 버퍼 길이 또는 10 중 작은 값
    #     # print(f"\n--- last_proprioceptive_bufs (Env 0, First {num_steps_to_print} Steps) ---")
    #     # # last_proprioceptive_bufs[시간 인덱스, 환경 인덱스, 관찰 인덱스]
    #     # # 인덱스 0이 최신, 1이 1스텝 전...
    #     # print(self.last_proprioceptive_bufs[:num_steps_to_print, 0, :]) # 첫 num_steps_to_print 스텝 (0번 스텝이 최신), 0번 환경, 모든 관찰 요소
    #     # print("-" * 30)
    #     # --- 디버깅 출력 끝 ---

    #     # 최종 지연된 proprioceptive 관찰 값을 저장할 텐서 생성
    #     num_proprioceptive_obs = self.last_proprioceptive_bufs.shape[-1] # 전체 proprioceptive 관찰 벡터의 크기
    #     delayed_proprioceptive_buf = torch.empty(
    #         (self.num_envs, num_proprioceptive_obs),
    #         dtype=self.last_proprioceptive_bufs.dtype,
    #         device=self.device
    #     )

    #     for obs_range_tuple, sensor_type in range_to_sensor_type.items():
    #         # 해당 범위에 적용할 지연 스텝 텐서 가져오기
    #         delay_frames_tensor = sensor_type_to_delay_frames.get(sensor_type)
    #         if delay_frames_tensor is None:
    #             print(f"Warning: Latency range for sensor type '{sensor_type}' not found in config. Defaulting to zero delay for range {obs_range_tuple}.")
    #             delay_frames_tensor = sensor_type_to_delay_frames["zero_delay"]

    #         # Create the slice object from the tuple
    #         obs_slice = slice(obs_range_tuple[0], obs_range_tuple[1])

    #         # --- 수정된 슬라이싱 로직 ---
    #         # 1단계: 시간 및 환경 차원에 대한 고급 인덱싱 수행
    #         # 결과 텐서의 형태는 (num_envs, num_proprioceptive_obs)가 됩니다.
    #         intermediate_result = self.last_proprioceptive_bufs[
    #             delay_frames_tensor, # 시간 차원 인덱스 (형태: num_envs,)
    #             torch.arange(self.num_envs, device=self.device) # 환경 차원 인덱스 (형태: num_envs,)
    #         ]

    #         # 2단계: 결과 텐서에 대해 관찰 차원 슬라이싱 수행
    #         # 결과 텐서의 형태는 (num_envs, end - start)가 됩니다.
    #         delayed_slice_data = intermediate_result[:, obs_slice].clone()
    #         # --- 수정된 슬라이싱 로직 끝 ---

    #         # 최종 지연된 proprioceptive 버퍼의 해당 구간에 데이터 할당
    #         delayed_proprioceptive_buf[:, obs_slice] = delayed_slice_data

    #     self.delayed_proprioceptive_buf = delayed_proprioceptive_buf


    #     # print("last_proprioceptive_buf : " , self.last_proprioceptive_bufs[:,0])
    #     # print("delayed_proprioceptive_buf : ", self.delayed_proprioceptive_buf[0])
    #     return self.delayed_proprioceptive_buf



    def compute_reward(self):
        self.foot_pos= self.rigid_body_pos[:,[7,14],:]
        foot_velocities = self.rigid_body_vel[:,[7,14],:]
        # self.standing = torch.norm(self.commands,dim=-1) == 0.
        self.no_commands = (torch.norm(self.commands,dim=-1) == 0)
        foot_contact_forces = self.contact_forces[:, [7,14], :]
        calf_contact_forces = torch.norm(self.contact_forces[:, [6,13], :],dim=-1)

        # foot_contact =  torch.any(self.contact_forces[:, self.feet_indices, :],dim=-1) > 1.
        self.foot_contact = torch.sum(torch.square(foot_contact_forces),dim=-1) > 1.
        #===========================================< Task Rewards (Pos) >==============================================    
        # velocity tracking reward
        only_yaw = (self.commands_yaw != 0) & (self.commands_x == 0) & (self.commands_y == 0)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.rew_scales["rew_ang_vel_z"]
        # rew_ang_vel_z = torch.where(only_yaw, rew_ang_vel_z*1.5, rew_ang_vel_z)

        lin_vel_error_xy = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2],dim=-1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error_xy/0.25) * self.rew_scales["rew_lin_vel_xy"] 
        # rew_lin_vel_xy = torch.where(only_yaw, rew_lin_vel_xy*0.5, rew_lin_vel_xy)

        target_air_time = 0.5
        foot_landing = ~self.last_foot_contacts & self.foot_contact
        air_time_difference = self.foot_air_time - target_air_time
        rew_feet_air_time = torch.sum(air_time_difference * foot_landing, dim=-1) * self.rew_scales["rew_feet_air_time"]
        self.foot_air_time[~self.foot_contact] += self.policy_dt 
        self.foot_air_time[self.foot_contact] = 0.
        self.last_foot_contacts = self.foot_contact.clone()

        #===========================================< Task Penalties (Neg) >==============================================
        
        penalty_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["penalty_lin_vel_z"]
        penalty_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["penalty_ang_vel_xy"]

        # base_ori penalty
        base_ori_err = torch.norm(self.projected_gravity[:,:2],dim=-1) 
        penalty_base_ori = base_ori_err * self.rew_scales["penalty_base_ori"]

        calf_contact = torch.norm(self.contact_forces[:, [4,5,11,12], :], dim=2) > 1.
        penalty_calf_contact= torch.sum(calf_contact, dim=1) * self.rew_scales["penalty_calf_contact"] # sum vs any ?

        # base height penalty
        base_height = self.base_pos[:,2]
        ref_body_height = 0.9 # ?
        base_height_err = abs((ref_body_height) - base_height)
        penalty_base_height = base_height_err * self.rew_scales["penalty_base_height"]        

        #Swing_phase
        cycle_lower = 0.25
        swing_phase = self.sin_cycle > cycle_lower
        stance_phase = self.sin_cycle < cycle_lower 
        foot_force = torch.norm(foot_contact_forces,dim=-1)
        foot_velocities_squared_sum= torch.sum(torch.square(foot_velocities),dim=-1)
        swing_stance_penalty = torch.clip(torch.sum(foot_velocities_squared_sum*stance_phase, dim=-1),0,10)*0.05 +\
                               torch.clip(torch.sum(foot_force*swing_phase, dim=-1),0,30)*0.025 
        
        penalty_swing_stance_phase = swing_stance_penalty * ~self.no_commands * self.rew_scales["penalty_swing_stance_phase"]

        # foot gap
        foot_gap_y = self.rigid_body_pos[:,7,1]-self.rigid_body_pos[:,14,1]
        ref_foot_gap_y = 0.125
        foot_gap_y_err= torch.relu(ref_foot_gap_y - foot_gap_y) 
        foot_gap_y_penalty = foot_gap_y_err * self.rew_scales["foot_gap_y_penalty"]

        foot_gap_x = abs(self.rigid_body_pos[:,7,0]-self.rigid_body_pos[:,14,0]) * self.no_commands
        foot_gap_x_penalty = foot_gap_x * self.rew_scales["foot_gap_x_penalty"]

        leg_gap_y = self.rigid_body_pos[:,[3,4],1]-self.rigid_body_pos[:,[10,11],1]
        ref_leg_gap_y = 0.1
        leg_gap_y_err= torch.sum(torch.relu(ref_leg_gap_y - leg_gap_y),dim=-1) 
        leg_gap_y_penalty = leg_gap_y_err * self.rew_scales["leg_gap_y_penalty"]


        # short swing penalty
        self.foot_swing_state[~self.foot_contact] = True
        self.foot_swing_state[self.foot_contact] = False
        
        current_time = self.gym.get_sim_time(self.sim)  # 현재 시뮬레이션 시간 0.01 0.02  
        self.foot_swing_start_time[~self.foot_contact] = current_time # 0.01 / 0.02
        swing_time = current_time - self.foot_swing_start_time
        contact = (swing_time == 0.)
        penalty_short_swing = torch.sum(torch.relu(self.min_swing_time - swing_time) * ~contact,dim=-1)* self.rew_scales["penalty_short_swing"]
    
        # penalty_foot_height
        # ref_foot_h = 0.07
        ref_foot_height = 0.075
        # min_foot_height = self.foot_pos[:,:,2] > ref_foot_height        
        foot_height_err = abs(self.foot_pos[:,:,2] - ref_foot_height) #* self.foot_contact # * torch.norm(foot_velocities[:,:2],dim=-1) #* ~min_foot_height
        penalty_foot_height = torch.sum(foot_height_err,dim=-1)*~self.no_commands *self.rew_scales["penalty_foot_height"]

        # penalty_default_pos_standing
        default_pos_err = torch.square(self.default_dof_pos - self.dof_pos)
        penalty_default_pos_standing = torch.sum(default_pos_err,dim=-1) * self.no_commands * self.rew_scales["penalty_default_pos_standing"]

        y_cmd_zero = self.commands_y == 0
        yaw_cmd_zero = self.commands_yaw == 0

        # hip yaw
        hip_yaw = torch.square(self.dof_pos[:, [0,6]])
        penalty_hip_yaw = torch.sum(hip_yaw,dim=-1) * self.rew_scales["penalty_hip_yaw"]
        # penalty_hip_yaw = torch.where(self.no_commands,
        #                                 penalty_hip_yaw * 2.,
        #                                 penalty_hip_yaw)

        # hip roll -> y나 yaw가 있어야만 쓴다.
        hip_roll = torch.square(self.dof_pos[:, [1,7]])
        penalty_hip_roll = torch.sum(hip_roll,dim=-1) * self.rew_scales["penalty_hip_roll"]* y_cmd_zero * yaw_cmd_zero
        # penalty_hip_roll = torch.where(self.no_commands,
        #                                 penalty_hip_roll * 2.,
        #                                 penalty_hip_roll)


        #ankle roll
        ankle_roll = torch.square(self.dof_pos[:, [5,11]])
        penalty_ankle_roll = torch.sum(ankle_roll,dim=-1) * self.rew_scales["penalty_ankle_roll"]* y_cmd_zero 
        # penalty_ankle_roll = torch.where(self.no_commands,
        #                                 penalty_ankle_roll * 2.,
        #                                 penalty_ankle_roll)
        
        #jump
        num_contact_feet = torch.sum(torch.norm(self.contact_forces[:, [4, 7], :],dim=-1) > 1., dim=-1)
        zero_contact_feet_err =torch.where((num_contact_feet == 0),
                        torch.ones_like(penalty_ankle_roll),
                        torch.zeros_like(penalty_ankle_roll))
        penalty_zero_contact_feet = zero_contact_feet_err * self.rew_scales["penalty_zero_contact_feet"]


        # mean_grf_foot = torch.mean(foot_contact_forces,dim=-1)
        # print("mean_grf : ", mean_grf)

        # mean_grf_feet = torch.mean(mean_grf_foot,dim=-1)
        # print("mean_grf_z : ", mean_grf_z)
        

        grf_err =  torch.sum(torch.abs((220 - foot_contact_forces[:,:,2])),dim=-1)
        penalty_default_grf = grf_err * self.rew_scales["penalty_default_grf"] * self.no_commands


        # dof_limits
        softness = 0.85 

        dof_pos_limit_lower = torch.sum(torch.relu(self.dof_pos_limits[:, 0].unsqueeze(0)*softness -self.dof_pos),dim=-1)
        dof_pos_limit_upper = torch.sum(torch.relu(self.dof_pos-self.dof_pos_limits[:, 1].unsqueeze(0)*softness),dim=-1)
        penalty_dof_limit = (dof_pos_limit_lower + dof_pos_limit_upper) * self.rew_scales["penalty_dof_limit"]
    
        actions_limit_lower = torch.sum(torch.relu(self.dof_pos_limits[:, 0].unsqueeze(0)*softness - self.actions),dim=-1)
        actions_limit_upper = torch.sum(torch.relu(self.actions-self.dof_pos_limits[:, 1].unsqueeze(0)*softness),dim=-1)
        penalty_actions_limit = (actions_limit_lower + actions_limit_upper) * self.rew_scales["penalty_actions_limit"]
    
        # joint_vel_limit
        dof_vel_limit_over = torch.relu(self.dof_vel - self.dof_vel_limits.unsqueeze(0))
        penalty_dof_vel_limit = torch.sum(dof_vel_limit_over, dim=-1) * self.rew_scales["penalty_dof_vel_limit"]

        # 
        mean_feet_pos_xy = torch.mean(self.foot_pos[:,:,:2],dim=1)
        robot_CoM_xy = self.robot_CoM[:,:2]
        com_sup_err = torch.norm(mean_feet_pos_xy - robot_CoM_xy, dim=-1)
        penalty_com_sup = com_sup_err * self.rew_scales["penalty_com_sup"]

        # mean(self.foot_pos[:,:,:2]

        # clock penalty 
        phi_err = abs(abs(self.cycle_L_x - self.cycle_R_x))
        penalty_phi = phi_err * self.rew_scales["penalty_phi"]




        #===========================================< Regulation Penalties (Neg) >==============================================
        # torque penalty
        penalty_torques = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["penalty_torques"]
        
        # joint vel penalty
        joint_vel = abs(self.dof_vel) 
        joint_vel = torch.sum(torch.square(joint_vel), dim=1)
        penalty_joint_vel = joint_vel * self.rew_scales["penalty_joint_vel"]


        # Energy penalty
        joint_energy = torch.sum(torch.square(self.torques * self.dof_vel),dim=-1)
        penalty_joint_energy = joint_energy * self.rew_scales["penalty_joint_energy"]

        # dof acc
        self.dof_acc = (self.dof_vel -self.last_dof_vel)
        joint_acc_err = torch.sum(torch.square(self.dof_acc),dim=-1)
        penalty_joint_acc = joint_acc_err * self.rew_scales["penalty_joint_acc"]

        # weighted torques
        wighted_torques = torch.sum(torch.square(self.torques/self.Kp),dim=-1)
        penalty_wighted_torques = wighted_torques * self.rew_scales["penalty_wighted_torques"]

        # penalty_action_smoothness1
        action_smoothness1_diff = torch.square(self.actions-self.last_actions)
        action_smoothness1_diff *= (self.last_actions != 0)
        penalty_action_smoothness1 = torch.sum(action_smoothness1_diff, dim=1) * self.rew_scales["penalty_action_smoothness1"]
        
        # penalty_action_smoothness2
        action_smoothness2_diff = torch.square(self.actions - 2 *self.last_actions + self.last_actions2)
        action_smoothness2_diff *= (self.last_actions != 0)
        action_smoothness2_diff *= (self.last_actions2 != 0)
        penalty_action_smoothness2 = torch.sum(action_smoothness2_diff, dim=1) * self.rew_scales["penalty_action_smoothness2"]

        
        #===========================================< Calculate Rewards >==============================================

        task_reward = rew_feet_air_time + rew_lin_vel_xy + rew_ang_vel_z #+rew_lin_vel_x + rew_lin_vel_y

        task_penalty = penalty_com_sup+ penalty_phi + leg_gap_y_penalty + penalty_ang_vel_xy + penalty_dof_vel_limit +  penalty_lin_vel_z + penalty_actions_limit + penalty_dof_limit + foot_gap_x_penalty + penalty_wighted_torques + penalty_joint_acc + penalty_joint_energy + penalty_short_swing+ penalty_default_grf+ foot_gap_y_penalty + penalty_calf_contact + penalty_zero_contact_feet+penalty_hip_yaw+ penalty_ankle_roll + penalty_hip_roll+ penalty_base_ori + penalty_base_height + penalty_swing_stance_phase + penalty_foot_height + penalty_default_pos_standing
        task_penalty += penalty_torques + penalty_joint_vel + penalty_action_smoothness1 + penalty_action_smoothness2

        total_rew = task_reward + task_penalty 
        # total_rew = task_reward 

        for key in self.cfg["env"]["learn"]["reward"].keys():
                self.reward_container[key] = locals()[key][self.observe_envs]
    
        # total reward
        self.rew_buf = total_rew
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

    def plot_juggler(self):
        base_lin_vel_x = self.base_lin_vel[self.observe_envs,0]
        base_lin_vel_y = self.base_lin_vel[self.observe_envs,1]
        base_ang_vel_z = self.base_ang_vel[self.observe_envs,2]

        commands_x  = self.commands_x[self.observe_envs]
        commands_y  = self.commands_y[self.observe_envs]
        commands_yaw  = self.commands_yaw[self.observe_envs]

        # commands        
        command_obs_msg = Twist()
        command_obs_msg.linear.x = commands_x
        command_obs_msg.linear.y = commands_y
        command_obs_msg.angular.z = commands_yaw

        # base_state
        twist_msg = Twist()
        twist_msg.linear.x = base_lin_vel_x
        twist_msg.linear.y = base_lin_vel_y
        twist_msg.angular.z = base_ang_vel_z


        #
        sin_L_msg = Float32()
        sin_L_msg.data = self.sin_cycle[self.observe_envs,0]
        sin_L_pub = rospy.Publisher(f"/sin_L", Float32, queue_size=10)
        sin_L_pub.publish(sin_L_msg)

        cos_L_msg = Float32()
        cos_L_msg.data = self.cos_cycle[self.observe_envs,0]
        cos_L_pub = rospy.Publisher(f"/cos_L", Float32, queue_size=10)
        cos_L_pub.publish(cos_L_msg)

        sin_R_msg = Float32()
        sin_R_msg.data = self.sin_cycle[self.observe_envs,1]
        sin_R_pub = rospy.Publisher(f"/sin_R", Float32, queue_size=10)
        sin_R_pub.publish(sin_R_msg)

        cos_R_msg = Float32()
        cos_R_msg.data = self.cos_cycle[self.observe_envs,1]
        cos_R_pub = rospy.Publisher(f"/cos_R", Float32, queue_size=10)
        cos_R_pub.publish(cos_R_msg)


        for reward_name, reward_value in self.reward_container.items():
            reward_msg = Float32()
            reward_msg.data = reward_value /self.policy_dt
            reward_pub = rospy.Publisher(f"/{reward_name}", Float32, queue_size=10)
            reward_pub.publish(reward_msg)

        tot_reward_msg = Float32()
        tot_reward_msg.data = self.rew_buf[self.observe_envs] / self.policy_dt
        tot_reward_pub = rospy.Publisher(f"/tot_reward", Float32, queue_size=10)
        tot_reward_pub.publish(tot_reward_msg)

        Joint_state = JointState()
        Joint_state.name = self.dof_names
        Joint_state.position = abs(self.dof_pos[self.observe_envs,:])
        Joint_state.velocity = abs(self.dof_vel[self.observe_envs,:])
        Joint_state.effort   = abs(self.torques[self.observe_envs,:])

        actions = JointState()
        actions.name = self.dof_names
        actions.position = self.actions[self.observe_envs,:]

        # ================================= publish =================================

        # ROS Publisher 생성
        command_obs_pub = rospy.Publisher('/command_obs_pub', Twist, queue_size=10)
        cmd_vel_obs_pub = rospy.Publisher('/cmd_vel_obs_pub', Twist, queue_size=10)

        # cycle_pub = rospy.Publisher('/cycle', Point, queue_size=10)
        #Joint_state
        joint_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        joint_state_pub.publish(Joint_state)

        #actions_pub
        actions_pub = rospy.Publisher('/isaacgym_actions', JointState, queue_size=10)
        actions_pub.publish(actions)

        # commands
        command_obs_pub.publish(command_obs_msg)
        cmd_vel_obs_pub.publish(twist_msg)


    
    def joy_callback(self, data):
        self.commands_x[:] = data.axes[1] * self.command_x_range[1]  # x vel
        self.commands_y[:] = data.axes[0] * self.command_y_range[1]  # y vel
        self.commands_yaw[:] = data.axes[3] * self.command_yaw_range[1]  # yaw vel

        self.need_reset = data.buttons[0] * data.buttons[1]

        if self.cam_change_flag == False:
            if self.cam_change_cnt < 50:
                self.cam_change_cnt += 1
            else:
                self.cam_change_flag = True
        
        if (self.cam_change_flag)&(data.buttons[4] == 1 and data.buttons[5] == 1):
            self.cam_mode = (self.cam_mode + 1) % 4  # 0, 1, 2, 3 순환
            if self.cam_mode == 0:
                print(f"fix_cam 상태 변경: {self.cam_mode} (자유 시점)")
            elif self.cam_mode == 1:
                print(f"fix_cam 상태 변경: {self.cam_mode} (고정 시점)")
            elif self.cam_mode == 2:
                print(f"fix_cam 상태 변경: {self.cam_mode} (1인칭 시점)")
            elif self.cam_mode == 3:
                print(f"fix_cam 상태 변경: {self.cam_mode} (3인칭 시점)")
            self.cam_change_flag = False
            self.cam_change_cnt = 0


        if data.buttons[3]:
            self.push_robots()
    
    def set_cmd(self,env_ids):
        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids),1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids),1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids),1), device=self.device).squeeze()

        self.commands_x[env_ids] = torch.where(torch.abs(self.commands_x[env_ids]) <= 0.05, torch.tensor(0.0, device=self.device), self.commands_x[env_ids])
        self.commands_y[env_ids] = torch.where(torch.abs(self.commands_y[env_ids]) <= 0.05, torch.tensor(0.0, device=self.device), self.commands_y[env_ids])
        self.commands_yaw[env_ids] = torch.where(torch.abs(self.commands_yaw[env_ids]) <= 0.05, torch.tensor(0.0, device=self.device), self.commands_yaw[env_ids])

        # 학습용 커맨드 
        no_command_env_ids = env_ids[(env_ids >= self.stand_env_range[0]) & (env_ids <= self.stand_env_range[1])]
        self.commands_x[no_command_env_ids] = 0.
        self.commands_y[no_command_env_ids] = 0.
        self.commands_yaw[no_command_env_ids] = 0. 
        
        # only_plus_x = env_ids[(env_ids >= self.only_plus_x_envs_range[0]) & (env_ids <= self.only_plus_x_envs_range[1])]
        # self.commands_x[only_plus_x] = torch_rand_float(0, self.command_x_range[1], (len(only_plus_x),1), device=self.device).squeeze()
        # self.commands_y[only_plus_x] = 0.
        # self.commands_yaw[only_plus_x] = 0. 
        # only_minus_x = env_ids[(env_ids >= self.only_minus_x_envs_range[0]) & (env_ids <= self.only_minus_x_envs_range[1])]  
        # self.commands_x[only_minus_x] = torch_rand_float(self.command_x_range[0], 0, (len(only_minus_x),1), device=self.device).squeeze()
        # self.commands_y[only_minus_x] = 0.
        # self.commands_yaw[only_minus_x] = 0. 

        only_plus_x = env_ids[(env_ids >= self.only_plus_x_envs_range[0]) & (env_ids <= self.only_plus_x_envs_range[1])]
        self.commands_x[only_plus_x] = torch_rand_float(0, self.command_x_range[1], (len(only_plus_x),1), device=self.device).squeeze()
        self.commands_y[only_plus_x] = 0.
        self.commands_yaw[only_plus_x] = 0. 
        only_minus_x = env_ids[(env_ids >= self.only_minus_x_envs_range[0]) & (env_ids <= self.only_minus_x_envs_range[1])]  
        self.commands_x[only_minus_x] = torch_rand_float(self.command_x_range[0], 0, (len(only_minus_x),1), device=self.device).squeeze()
        self.commands_y[only_minus_x] = 0.
        self.commands_yaw[only_minus_x] = 0. 

        only_plus_y = env_ids[(env_ids >= self.only_plus_y_envs_range[0]) & (env_ids <= self.only_plus_y_envs_range[1])]
        self.commands_x[only_plus_y] = 0.
        self.commands_y[only_plus_y] = self.command_y_range[1]
        self.commands_yaw[only_plus_y] = 0. 
        only_minus_y = env_ids[(env_ids >= self.only_minus_y_envs_range[0]) & (env_ids <= self.only_minus_y_envs_range[1])]
        self.commands_x[only_minus_y] = 0.
        self.commands_y[only_minus_y] = self.command_y_range[0]
        self.commands_yaw[only_minus_y] = 0. 

        only_plus_yaw = env_ids[(env_ids >= self.only_plus_yaw_envs_range[0]) & (env_ids <= self.only_plus_yaw_envs_range[1])]
        self.commands_x[only_plus_yaw] = 0.
        self.commands_y[only_plus_yaw] = 0.
        self.commands_yaw[only_plus_yaw] = self.command_yaw_range[1]
        only_minus_yaw = env_ids[(env_ids >= self.only_minus_yaw_envs_range[0]) & (env_ids <= self.only_minus_yaw_envs_range[1])]
        self.commands_x[only_minus_yaw] = 0.
        self.commands_y[only_minus_yaw] = 0.
        self.commands_yaw[only_minus_yaw] = self.command_yaw_range[0]

        plus_x_plus_yaw_envs_range = env_ids[(env_ids >= self.plus_x_plus_yaw_envs_range[0]) & (env_ids <= self.plus_x_plus_yaw_envs_range[1])]
        self.commands_x[plus_x_plus_yaw_envs_range] = self.command_x_range[1]
        self.commands_y[plus_x_plus_yaw_envs_range] = 0.
        self.commands_yaw[plus_x_plus_yaw_envs_range] = self.command_yaw_range[1]
        plus_x_minus_yaw_envs_range = env_ids[(env_ids >= self.plus_x_minus_yaw_envs_range[0]) & (env_ids <= self.plus_x_minus_yaw_envs_range[1])]
        self.commands_x[plus_x_minus_yaw_envs_range] = self.command_x_range[1]
        self.commands_y[plus_x_minus_yaw_envs_range] = 0.
        self.commands_yaw[plus_x_minus_yaw_envs_range] = self.command_yaw_range[0]

        minus_x_plus_yaw_envs_range = env_ids[(env_ids >= self.minus_x_plus_yaw_envs_range[0]) & (env_ids <= self.minus_x_plus_yaw_envs_range[1])]
        self.commands_x[minus_x_plus_yaw_envs_range] = self.command_x_range[0]
        self.commands_y[minus_x_plus_yaw_envs_range] = 0.
        self.commands_yaw[minus_x_plus_yaw_envs_range] = self.command_yaw_range[1]
        minus_x_minus_yaw_envs_range = env_ids[(env_ids >= self.minus_x_minus_yaw_envs_range[0]) & (env_ids <= self.minus_x_minus_yaw_envs_range[1])]
        self.commands_x[minus_x_minus_yaw_envs_range] = self.command_x_range[0]
        self.commands_y[minus_x_minus_yaw_envs_range] = 0.
        self.commands_yaw[minus_x_minus_yaw_envs_range] = self.command_yaw_range[0]

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.925, 1.075, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]  * positions_offset
        self.dof_vel[env_ids] = velocities

        self.body_mass_noise[env_ids] = torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_bodies), device=self.device)
        self.body_mass[env_ids]       = self.robot_config_buffer[env_ids,:,3] + self.body_mass_noise[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.25, 0.25, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.termination_rew(env_ids)
        self.set_cmd(env_ids)
    
        self.last_foot_contacts[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        self.last_clock_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_actions2[env_ids] = 0.

        self.last_dof_vel[env_ids] = 0.
        self.last_dof_vel2[env_ids] = 0.

        self.dof_acc[env_ids] = 0.
        self.last_dof_acc[env_ids] = 0.

        self.cycle_t[env_ids,:] = 0.

        # for i in range(len(self.last_proprioceptive_bufs) - 1, 0, -1):
        #     self.last_proprioceptive_bufs[i][env_ids] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def termination_rew(self, env_ids):
        dof_limit_over = self.dof_limit_lower | self.dof_limit_upper | self.dof_vel_over
        self.rew_buf[env_ids] += self.progress_buf[env_ids] * 0.0001
        self.rew_buf[env_ids] = torch.where(~self.time_out[env_ids], self.rew_buf[env_ids]*0.25, self.rew_buf[env_ids])

        # print("self.time_out : ", self.time_out.size())
        
    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.rew_buf[env_ids] += self.terrain_levels[env_ids] * 0.1
        self.rew_buf[env_ids] += distance * 0.1
        # self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.1)

        # self.terrain_levels[env_ids] = torch.where((distance >= self.terrain.env_length/2 * ~self.no_commands[env_ids]) | self.time_out[env_ids],
        #                                            self.terrain_levels[env_ids] + 1,
        #                                            self.terrain_levels[env_ids])
        
        non_time_out_envs = env_ids[~self.time_out[env_ids]]
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[non_time_out_envs] -= 1
        # self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.25)
        # self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows


        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],
                                                            self.terrain_types[env_ids]]
        
    def push_robots(self):
        self.root_states[:, [7,8]] = torch_rand_float(-0.5, 0.5, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, [10,11,12]] = torch_rand_float(-0.5, 0.5, (self.num_envs, 3), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def pre_physics_step(self, actions):
        self.actions = actions[:,:13].clone().to(self.device) * self.action_scale
        self.clock_actions = actions[:,13:].clone().to(self.device)
        
        self.clock_actions[:,:2]  *= 1.  
        self.clock_actions[:,2:4] *= 0.025
        self.clock_actions[:,2:4] = torch.abs(self.clock_actions[:,2:4])

        phi_gain = 0.5
        delta_gain = 0.5
        self.clock_actions[:,:2]  = self.last_clock_actions[:,:2] * (1-phi_gain)+ self.clock_actions[:,:2] * phi_gain
        self.clock_actions[:,2:4] = self.last_clock_actions[:,2:4] * (1-delta_gain) + self.clock_actions[:,2:4] * delta_gain

        for _ in range(self.decimation-1):
            scaled_actions = self.action_scale * self.actions    
            targets = self.default_dof_pos.clone()  # 기본 관절 각도로 초기화
            targets += scaled_actions
            self.targets = targets.clone()
            torques = self.Kp*(targets - self.dof_pos) - self.Kd*self.dof_vel
            # torques = self.Kp*(self.default_dof_pos - self.dof_pos) - self.Kd*self.dof_vel
            torques = torch.clip(torques,-self.joint_torque_limits,self.joint_torque_limits)
            # print("torques1 : ", self.torques[0])
            # print("torques2 : ", torques[0])

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques = torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            # self.refresh_state()
            
            # # self.compute_observations_step()
            # # print("proprioceptive_buf : ", self.proprioceptive_buf[0])
            # for i in range(len(self.last_proprioceptive_bufs) - 1, 0, -1):
            #     self.last_proprioceptive_bufs[i][:] = self.last_proprioceptive_bufs[i - 1][:]
            
            # self.last_proprioceptive_bufs[0][:] = self.proprioceptive_buf[:]            

    
    def post_physics_step(self):
        self.refresh_state()
        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        # print(self.cycle_t.size())
        # print(self.clock_actions.size())
        self.cycle_t[:,0] += self.clock_actions[:,2]
        self.cycle_t[:,1] += self.clock_actions[:,3]

        self.no_commands = (torch.norm(self.commands,dim=-1) == 0)
        self.cycle_t *= ~self.no_commands.unsqueeze(1)
        # self.cycle_t = torch.where(torch.norm(self.commands,dim=-1) == 0.,
        #                             self.cycle_t * 0.,
        #                             self.cycle_t
        #                             )

        self.cycle_t = torch.where(self.cycle_t > self.cycle_time,
                                                        self.cycle_t - self.cycle_time,
                                                        self.cycle_t)
        
        if (self.common_step_counter % self.push_interval == 0) and not self.test_mode:
            self.push_robots()

        if self.common_step_counter % self.freeze_interval == 0 and not self.test_mode:
            self.freeze()
            self.freeze_flag = True
            self.freeze_steps = random.randint(150, 300)
        
        all_env_ids = torch.arange(self.num_envs)
        if self.freeze_flag:
            self.freeze_cnt += 1
            if self.freeze_cnt >= self.freeze_steps:
                self.set_cmd(all_env_ids)
                
                self.freeze_flag=False
                self.freeze_cnt = 0




        # self.measured_heights, self.measured_legs_heights = self.get_heights()
        # self.mean_measured_heights = torch.mean(self.measured_heights, dim=-1)
        # self.max_measured_legs_heights = torch.max(self.measured_legs_heights, dim=-1).values
        # self.min_measured_legs_heights = torch.min(self.measured_legs_heights, dim=-1).values
        # self.mean_measured_legs_heights = torch.mean(self.measured_legs_heights, dim=-1)
        
        # print("===========================================================")
        # # print("self.mean_measured_heights: ", self.mean_measured_heights[self.observe_envs])
        # print("self.mlh size : ", self.measured_legs_heights.size())
        # print("self.measured_legs_heights: ",self.measured_legs_heights[self.observe_envs])
        # print("self.max_measured_legs_heights: ", self.max_measured_legs_heights[self.observe_envs])
        # print("===========================================================")


        # compute observations, rewards, resets, ...
        self.reset_buf[:] = self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        # print("fsdata : ", self.fsdata[self.observe_envs])
        # print("contact_force : ", self.contact_forces[self.observe_envs,[7,14],:])
        # print("================================================")
        # print("progress_buf : ", self.progress_buf[self.observe_envs])
        # print("obs_t : ", self.obs_buf_t[self.observe_envs])
        # print("obs_buf : ", self.obs_buf[self.observe_envs])
        # print("================================================")
        self.plot_juggler()

        self.last_actions2[:] = self.last_actions[:]
        self.last_actions =  self.actions[:]

        self.last_clock_actions[:] = self.clock_actions[:] 

        self.last_dof_vel2[:] = self.last_dof_vel[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        
        self.last_dof_acc[:] = self.dof_acc[:]

        # if self.cam_mode != 0:        
        #     self.camera_update()

    def camera_update(self):
        if (self.test_mode) & (self.graphics_device_id != -1):
            offset = 1.5 #[m]

            base_pos = self.root_states[self.observe_envs, :3]

            if self.cam_mode == 1:
                cam_pos = gymapi.Vec3(base_pos[0] + offset, base_pos[1] + offset, base_pos[2] + offset)
                cam_target = gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2])

            elif self.cam_mode == 2:
                # 1인칭 시점: 카메라 위치를 로봇 베이스 위치 + offset으로 설정
                cam_pos_relative = gymapi.Vec3(0, 0, offset/3)
                cam_pos_relative_tensor = torch.tensor([cam_pos_relative.x, cam_pos_relative.y, cam_pos_relative.z], device=self.device).unsqueeze(0)
                rotated_offset = my_quat_rotate(self.base_quat, cam_pos_relative_tensor)
                rotated_offset_numpy = rotated_offset.cpu().numpy().flatten()
                cam_pos = gymapi.Vec3(*(base_pos.cpu().numpy() + rotated_offset_numpy))
                
                cam_target_relative = gymapi.Vec3(2*offset, 0, 0)
                cam_target_relative_tensor = torch.tensor([cam_target_relative.x, cam_target_relative.y, cam_target_relative.z], device=self.device).unsqueeze(0)
                rotated_offset_target = my_quat_rotate(self.base_quat, cam_target_relative_tensor)
                rotated_offset_target_numpy = rotated_offset_target.cpu().numpy().flatten()
                cam_target = gymapi.Vec3(*(base_pos.cpu().numpy() + rotated_offset_target_numpy))

            elif self.cam_mode == 3:
                # 3인칭 시점: 로봇 뒤쪽 위에서 바라보는 시점
                cam_pos_relative = gymapi.Vec3(-2*offset, 0, offset)
                cam_pos_relative_tensor = torch.tensor([cam_pos_relative.x, cam_pos_relative.y, cam_pos_relative.z], device=self.device).unsqueeze(0)
                rotated_offset = my_quat_rotate(self.base_quat, cam_pos_relative_tensor)
                rotated_offset_numpy = rotated_offset.cpu().numpy().flatten()
                cam_pos = gymapi.Vec3(*(base_pos.cpu().numpy() + rotated_offset_numpy))
                
                cam_target_relative = gymapi.Vec3(2*offset, 0, 0)
                cam_target_relative_tensor = torch.tensor([cam_target_relative.x, 0, 0], device=self.device).unsqueeze(0)
                rotated_offset_target = my_quat_rotate(self.base_quat, cam_target_relative_tensor)
                rotated_offset_target_numpy = rotated_offset_target.cpu().numpy().flatten()
                cam_target = gymapi.Vec3(*(base_pos.cpu().numpy() + rotated_offset_target_numpy))


            if self.smoothed_cam_pos is None:
                self.smoothed_cam_pos = cam_pos
            else:
                self.smoothed_cam_pos = gymapi.Vec3(
                    self.smoothed_cam_pos.x * (1 - self.smoothing_alpha) + cam_pos.x * self.smoothing_alpha,
                    self.smoothed_cam_pos.y * (1 - self.smoothing_alpha) + cam_pos.y * self.smoothing_alpha,
                    self.smoothed_cam_pos.z * (1 - self.smoothing_alpha) + cam_pos.z * self.smoothing_alpha
                )

            if self.smoothed_cam_target is None:
                self.smoothed_cam_target = cam_target
            else:
                self.smoothed_cam_target = gymapi.Vec3(
                    self.smoothed_cam_target.x * (1 - self.smoothing_alpha) + cam_target.x * self.smoothing_alpha,
                    self.smoothed_cam_target.y * (1 - self.smoothing_alpha) + cam_target.y * self.smoothing_alpha,
                    self.smoothed_cam_target.z * (1 - self.smoothing_alpha) + cam_target.z * self.smoothing_alpha
                )

            self.gym.viewer_camera_look_at(self.viewer, None, self.smoothed_cam_pos, self.smoothed_cam_target)
 
    def freeze(self):
        self.commands_x[:] = torch.zeros(self.num_envs, device=self.device)
        self.commands_y[:] = torch.zeros(self.num_envs, device=self.device)
        self.commands_yaw[:] = torch.zeros(self.num_envs, device=self.device)

    def refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # # [뎁스 카메라 추가] 그래픽스 및 결과 동기화 (카메라 데이터 접근 전)
        # self.gym.fetch_results(self.sim, True) # True: 동기화 대기
        # self.gym.step_graphics(self.sim)       # 그래픽스 관련 업데이트 (카메라 렌더링 포함)
        # self.gym.render_all_camera_sensors(self.sim)
        # self.gym.start_access_image_tensors(self.sim)
        
        # # env_id = self.num_envs - 1
        # img_tensor = self.cam_tensors[0]
        # img = img_tensor.clone().cpu().numpy()
        # depth_img_np = img_tensor.clone().cpu().numpy() # (height, width)
        # self.gym.end_access_image_tensors(self.sim)

        # if self.debug_img:
            # _img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            # cv2.imshow("render", _img)
            # if cv2.waitKey(1) == 27:  # ESC
            #     exit()

        # if self.debug_img:
        #     near = self.camera_props.near_plane  # self.cfg["env"]["camera"]["near_plane"]
        #     far  = self.camera_props.far_plane   # self.cfg["env"]["camera"]["far_plane"]
            
        #     depth_img_np[depth_img_np == -np.inf] = 0
        #     depth_img_np[depth_img_np < -far] = -far

        #     # depth_normalized = np.clip(depth_normalized, 0, 1) # 0~1 사이로 클리핑
        #     depth_normalized = (depth_img_np - near) / (far - near)
            
        #     normalized_depth = -255.0 * (depth_img_np/np.min(depth_img_np + 1e-4))

        #     # 0-255 범위로 스케일링하고 uint8 타입으로 변환
        #     depth_display = (depth_normalized*255.).astype(np.uint8)

        #     # 3. 이미지 표시
        #     cv2.imshow("Depth Image Env 0 (OpenCV)", depth_display)

        #     if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
        #         # exit() # 전체 프로그램 종료 대신, 루프를 빠져나가거나 플래그 변경 등을 고려
        #         self.debug_img = False # 예시: 디버그 이미지 표시 중단

        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.last_base_vel = self.base_vel
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_vel = torch.cat([self.base_lin_vel, self.base_ang_vel], dim=1)


    def init_height_points(self):
        # 60cm x 25cm rectangle 
        y = 0.05 * torch.tensor([-5, -3,-1,1, 3, 5], device=self.device, requires_grad=False)
        x = 0.05 * torch.tensor([-12, -9, -6, -3, 3, 6, 9, 12], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_leg_height_points(self):
        """각 다리 아래의 높이를 측정할 지점들을 초기화합니다.

        각 다리 밑의 작은 사각형 영역 (예: 10cm x 10cm) 내에 균등하게 분포된 점들을 생성합니다.
        이 점들은 지면과의 거리를 측정하거나 다리의 높이를 추정하는 데 사용될 수 있습니다.

        Returns:
            torch.Tensor: (num_envs, 4, num_leg_height_points, 3) 크기의 텐서.
                        각 환경, 각 다리 아래의 측정 지점들의 (x, y, z) 좌표를 담고 있습니다.
        """
        # 각 다리 아래의 작은 사각형 범위 (예: 10cm x 10cm)를 정의하기 위한 리스트
        leg_points = []
        # 각 다리의 중심 위치를 기준으로 하는 오프셋 (상대적인 위치)
        # FL: Front Left (앞 왼쪽), FR: Front Right (앞 오른쪽),
        # RL: Rear Left (뒤 왼쪽), RR: Rear Right (뒤 오른쪽)
        leg_offsets = [
            (-0. , 0.), # FL (2사분면) - 수정: 의미상 약간 더 자연스러운 위치
            ( 0. , 0.),  # FR (1사분면) - 수정: 의미상 약간 더 자연스러운 위치
            (-0. , 0.),# RL (3사분면) - 수정: 의미상 약간 더 자연스러운 위치
            ( 0. , 0.)  # RR (4사분면) - 수정: 의미상 약간 더 자연스러운 위치
        ]

        # 각 다리 밑의 격자(grid) 형태로 점들을 생성
        # -2, -1, 0, 1, 2를 0.025 (2.5cm)씩 곱하여 -5cm, -2.5cm, 0cm, 2.5cm, 5cm 간격의 x, y 좌표 생성
        leg_grid_x, leg_grid_y = torch.meshgrid(
            0.02 * torch.tensor([-5, -3, -1, 1, 3, 5], device=self.device, requires_grad=False),
            0.02 * torch.tensor([-5, -3, -1, 1, 3, 5], device=self.device, requires_grad=False)
        )
        # 생성된 x, y 좌표들을 평탄화(flatten)하여 각 점의 (x, y) 쌍을 만듦
        leg_points_grid = torch.stack([leg_grid_x.flatten(), leg_grid_y.flatten()], dim=-1) # (25, 2) 크기

        # 각 다리에 대해 생성된 격자 점들을 오프셋만큼 이동시키고 z 좌표를 0으로 설정
        for offset_x, offset_y in leg_offsets:
            # 각 다리의 중심 오프셋에 격자 점들의 x, y 좌표를 더하고 z 좌표 0을 추가
            leg_points.append(torch.tensor([offset_x, offset_y, 0.0], device=self.device) + torch.cat([leg_points_grid, torch.zeros_like(leg_points_grid[:, 0:1])], dim=-1))

        # 각 다리 아래의 생성된 점들의 개수를 저장 (여기서는 5x5 = 25개)
        self.num_leg_height_points = leg_points_grid.shape[0]
        # 생성된 각 다리의 점들 리스트를 텐서로 쌓음 (4개의 다리, 각 25개의 점, 각 점은 [x, y, z] 좌표)
        leg_points_tensor = torch.stack(leg_points, dim=0) # (4, 25, 3) 크기
        # 환경의 개수만큼 복사하고 차원을 추가하여 최종적인 형태 (num_envs, 4, 25, 3)를 만듦
        leg_points_tensor = leg_points_tensor.unsqueeze(0).repeat(self.num_envs, 1, 1, 1)

        return leg_points_tensor
    
    def init_leg_height_points_radial(self):
        """각 다리 아래의 높이를 측정할 지점들을 초기화합니다 (방사형)."""
        leg_points = []
        leg_offsets = [
            (-0. , 0.),
            ( 0. , 0.),
            (-0. , 0.),
            ( 0. , 0.)
        ]

        num_radii = 2
        num_angles = 12
        radii = torch.tensor([0.025, 0.05], device=self.device, requires_grad=False) # 2.5cm, 5cm
        angles = torch.linspace(0, 2 * torch.pi, num_angles, device=self.device, requires_grad=False)

        points_local = torch.zeros((num_radii * num_angles + 1, 3), device=self.device)
        points_local[0] = torch.tensor([0.0, 0.0, 0.0], device=self.device) # 중심점

        for i in range(num_radii):
            for j in range(num_angles):
                angle = angles[j]
                radius = radii[i]
                x = radius * torch.cos(angle)
                y = radius * torch.sin(angle)
                points_local[i * num_angles + j + 1] = torch.tensor([x, y, 0.0], device=self.device)

        self.num_leg_height_points = points_local.shape[0]

        for offset_x, offset_y in leg_offsets:
            leg_points.append(torch.tensor([offset_x, offset_y, 0.0], device=self.device) + points_local)

        leg_points_tensor = torch.stack(leg_points, dim=0)
        leg_points_tensor = leg_points_tensor.unsqueeze(0).repeat(self.num_envs, 1, 1, 1)

        return leg_points_tensor
    
    def init_leg_height_points_sector(self):
        """각 다리 아래의 높이를 측정할 지점들을 초기화합니다 (부채꼴 모양, +x 방향, 10개 포인트)."""
        leg_points = []
        leg_offsets = [
            (-0. , 0.),
            ( 0. , 0.),
            (-0. , 0.),
            ( 0. , 0.)
        ]

        num_points = 15      # 중심점을 제외한 부채꼴 내부 포인트 수 (총 10개)
        min_radius = 0.05   # 최소 반지름
        max_radius = 0.20   # 최대 반지름
        angle_extent = torch.pi * 3 / 4  # 부채꼴의 각도 범위 (90도)
        angle_offset = 0.0            # 부채꼴의 시작 각도 (현재 +x 방향)

        points_local_list = []
        points_local_list.append(torch.tensor([0.0, 0.0, 0.0], device=self.device)) # 중심점

        # 포인트 수를 기반으로 반지름 및 각도 간격 조정
        if num_points > 0:
            # 대략적인 반지름 및 각도 분할 수 계산
            num_radii_approx = 3
            num_angles_approx = (num_points + num_radii_approx - 1) // num_radii_approx

            radii = torch.linspace(min_radius, max_radius, num_radii_approx, device=self.device)
            angles = torch.linspace(-angle_extent / 2 + angle_offset, angle_extent / 2 + angle_offset, num_angles_approx, device=self.device)

            count = 0
            for i in range(num_radii_approx):
                radius = radii[i]
                for j in range(num_angles_approx):
                    if count < num_points:
                        angle = angles[j]
                        x = radius * torch.cos(angle)
                        y = radius * torch.sin(angle)
                        points_local_list.append(torch.tensor([x, y, 0.0], device=self.device))
                        count += 1
                    else:
                        break

        points_local = torch.stack(points_local_list, dim=0)
        self.num_leg_height_points = points_local.shape[0]

        for offset_x, offset_y in leg_offsets:
            leg_points.append(torch.tensor([offset_x, offset_y, 0.0], device=self.device) + points_local)

        leg_points_tensor = torch.stack(leg_points, dim=0)
        leg_points_tensor = leg_points_tensor.unsqueeze(0).repeat(self.num_envs, 1, 1, 1)

        return leg_points_tensor
                
    def get_heights(self, env_ids=None):
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False), torch.zeros(self.num_envs, 4, self.num_leg_height_points, device=self.device, requires_grad=False)

        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids is not None:
            base_quat = self.base_quat[env_ids]
            root_states = self.root_states[env_ids]
            height_points = self.height_points[env_ids]
            leg_height_points = self.leg_height_points[env_ids]
            foot_quat = self.rigid_body_rot[env_ids][:, [7,14], :]
            foot_states = self.rigid_body_pos[env_ids][:, [7,14], :]
        else:
            base_quat = self.base_quat
            root_states = self.root_states
            height_points = self.height_points
            leg_height_points = self.leg_height_points
            foot_quat = self.rigid_body_rot[:, [7,14], :]
            foot_states = self.rigid_body_pos[:, [7,14], :]

        # 기존의 넓은 범위 높이 측정 (유지)
        points = quat_apply_yaw(base_quat.repeat(1, self.num_height_points), height_points) + (root_states[:, :3]).unsqueeze(1)
        world_x = points[:, :, 0]# 각 점의 월드 좌표계 x 좌표
        world_y = points[:, :, 1]  # 각 점의 월드 좌표계 y 좌표
        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)
        measured_heights = heights.view(root_states.shape[0], -1) * self.terrain.vertical_scale

        
        #p1, p2, p3 정의
        world_x -= self.terrain.border_size
        world_y -= self.terrain.border_size
        p1 = torch.stack([world_x[:, 0], world_y[:, 0], measured_heights[:, 0]], dim=-1)
        p2 = torch.stack([world_x[:, 6], world_y[:, 6], measured_heights[:, 6]], dim=-1)
        p3 = torch.stack([world_x[:, 11], world_y[:, 11], measured_heights[:, 11]], dim=-1)

        # print("p1  : ", p1)
        # print("p2  : ", p2)
        # print("p3  : ", p3)
        v1 = p2 - p1
        v2 = p3 - p1

        # 두 벡터의 외적 계산
        normal = torch.cross(v1, v2)

        # 법선 벡터 정규화
        normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True)

        
        
        # 각 발(Foot) 아래 높이 측정 (TIP 기준)
        num_envs = root_states.shape[0]
        num_legs = 2

        # leg_height_points (로컬 좌표) - 이미 (num_envs, 4, 25, 3) 크기임
        foot_points_local = leg_height_points

        # 발의 회전 적용 (전체 쿼터니언 사용)
        foot_points_rotated = quat_rotate(foot_quat.unsqueeze(2).repeat(1, 1, self.num_leg_height_points, 1).view(-1, 2),
                                          foot_points_local.view(-1, 3)).view(num_envs, num_legs, self.num_leg_height_points, 3)

        # 발의 위치 적용 (월드 좌표계)
        foot_points_world = foot_points_rotated + foot_states.unsqueeze(2).repeat(1, 1, self.num_leg_height_points, 1)

        # 월드 좌표를 terrain 맵 좌표로 변환
        foot_points_world += self.terrain.border_size
        foot_points_world_scaled = (foot_points_world / self.terrain.horizontal_scale).long()

        # terrain 맵 좌표를 사용하여 높이 샘플링
        fx = foot_points_world_scaled[:, :, :, 0].view(-1)
        fy = foot_points_world_scaled[:, :, :, 1].view(-1)

        fx = torch.clip(fx, 0, self.height_samples.shape[0] - 2)
        fy = torch.clip(fy, 0, self.height_samples.shape[1] - 2)

        foot_heights1 = self.height_samples[fx, fy]
        foot_heights2 = self.height_samples[fx + 1, fy + 1]
        foot_heights = torch.min(foot_heights1, foot_heights2)
        measured_foot_heights = foot_heights.view(num_envs, num_legs, self.num_leg_height_points) * self.terrain.vertical_scale


       # ========================================= Base 시각화 =========================================
        if self.test_mode:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom_base = gymutil.WireframeSphereGeometry(0.02, 2, 2, None, color=(0, 0, 1)) # base 측정 지점 색상 (파란색)

            for i in range(self.num_envs):
                # Base 시각화 (기존 코드)
                base_position = self.root_states[i, :3]
                base_rotation = self.base_quat[i]
                height_points_local_base = self.height_points[i].cpu().numpy() # base의 로컬 측정 포인트
                measured_heights_cpu_base = measured_heights[i].cpu().numpy() # base에서 측정된 높이

                for j in range(height_points_local_base.shape[0]):
                    local_point = torch.tensor(height_points_local_base[j], device=self.device)
                    world_point = quat_apply_yaw(base_rotation, local_point) + base_position
                    z = measured_heights_cpu_base[j]

                    sphere_pose = gymapi.Transform(gymapi.Vec3(world_point[0].item(), world_point[1].item(), z.item()), r=None)
                    gymutil.draw_lines(sphere_geom_base, self.gym, self.viewer, self.envs[i], sphere_pose)

        return measured_heights, measured_foot_heights

        # num_envs = root_states.shape[0]
        # num_legs = 4

        # # leg_height_points (로컬 좌표, 부채꼴 형태)
        # foot_points_local = leg_height_points

        # # 선형 속도 벡터로부터 방향 각도 계산
        # velocity_angles = torch.atan2(self.commands_y, self.commands_x)
        # zeros = torch.zeros_like(velocity_angles)
        # velocity_rotations_yaw_only = quat_from_euler_xyz(zeros, zeros, velocity_angles)
        # velocity_rotations = velocity_rotations_yaw_only.unsqueeze(1).repeat(1, num_legs, 1) # (num_envs, 4, 4)

        # # Yaw 커맨드 값
        # yaw_command = self.commands_yaw if hasattr(self, 'commands_yaw') else torch.zeros_like(velocity_angles)

        # # 선형 속도 크기 계산
        # linear_velocity_magnitude = torch.sqrt(self.commands_x**2 + self.commands_y**2)

        # # Yaw 회전 오프셋 계산 및 적용 조건
        # front_yaw_offset = yaw_command * 1.
        # rear_yaw_offset = -yaw_command * 1.
        # yaw_offsets = torch.stack([front_yaw_offset, front_yaw_offset , rear_yaw_offset,  rear_yaw_offset], dim=1) # [num_envs, num_legs]
        # yaw_rotations = quat_from_euler_xyz(zeros.unsqueeze(1).repeat(1, num_legs), zeros.unsqueeze(1).repeat(1, num_legs), yaw_offsets) # [num_envs, num_legs, 4]

        # combined_quat = torch.zeros(num_envs, num_legs, 4, device=self.device)
        # identity_quat = torch.tensor([0., 0., 0., 1.], device=self.device).unsqueeze(0).unsqueeze(0).repeat(num_envs, num_legs, 1)

        # for i in range(num_envs):
        #     if linear_velocity_magnitude[i] > 1e-2: # 임계값 설정 (조정 필요)
        #         # 선형 속도가 충분히 크면 속도 방향 사용
        #         combined_quat[i] = quat_mul(velocity_rotations[i].view(-1, 4), foot_quat[i].view(-1, 4)).view(num_legs, 4)
        #     else:
        #         # 선형 속도가 작으면 Yaw 커맨드 적용 (velocity_rotations 영향 없앰)
        #         combined_quat[i] = quat_mul(yaw_rotations[i].view(-1, 4), foot_quat[i].view(-1, 4)).view(num_legs, 4)

        # # 부채꼴 점들을 발의 위치로 이동시키기 전에, 결합된 회전 적용
        # foot_points_rotated = quat_rotate(combined_quat.unsqueeze(2).repeat(1, 1, self.num_leg_height_points, 1).view(-1, 4),
        #                                 foot_points_local.view(-1, 3)).view(num_envs, num_legs, self.num_leg_height_points, 3)


        # # 발의 위치 적용 (월드 좌표계)
        # foot_points_world = foot_points_rotated + foot_states.unsqueeze(2).repeat(1, 1, self.num_leg_height_points, 1)

        # # 월드 좌표를 terrain 맵 좌표로 변환
        # foot_points_world += self.terrain.border_size
        # foot_points_world_scaled = (foot_points_world / self.terrain.horizontal_scale).long()

        # # terrain 맵 좌표를 사용하여 높이 샘플링
        # fx = foot_points_world_scaled[:, :, :, 0].view(-1)
        # fy = foot_points_world_scaled[:, :, :, 1].view(-1)

        # fx = torch.clip(fx, 0, self.height_samples.shape[0] - 2)
        # fy = torch.clip(fy, 0, self.height_samples.shape[1] - 2)

        # foot_heights1 = self.height_samples[fx, fy]
        # foot_heights2 = self.height_samples[fx + 1, fy + 1]
        # foot_heights = torch.min(foot_heights1, foot_heights2)
        # measured_foot_heights = foot_heights.view(num_envs, num_legs, self.num_leg_height_points) * self.terrain.vertical_scale

        # # 시각화 (test_mode 활성화 시)
        # if self.test_mode:
        #     self.gym.clear_lines(self.viewer)
        #     self.gym.refresh_rigid_body_state_tensor(self.sim)
        #     sphere_geom_foot = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(1, 0, 0))
        #     for i in range(self.num_envs):
        #         foot_base_positions = self.rigid_body_pos[i, [7,14], :3]
        #         combined_rotation = combined_quat[i]
        #         leg_height_points_local = self.leg_height_points[i].cpu().numpy()
        #         measured_foot_heights_cpu = measured_foot_heights[i].cpu().numpy()

        #         for k in range(num_legs):
        #             base_pos = foot_base_positions[k]
        #             rotation = combined_rotation[k]
        #             heights = measured_foot_heights_cpu[k]
        #             local_points = leg_height_points_local[k]

        #             for j in range(local_points.shape[0]):
        #                 local_point = torch.tensor(local_points[j], device=self.device)
        #                 world_point = quat_apply_yaw(rotation, local_point) + base_pos
        #                 z = heights[j]

        #                 sphere_pose = gymapi.Transform(gymapi.Vec3(world_point[0].item(), world_point[1].item(), z.item()), r=None)
        #                 gymutil.draw_lines(sphere_geom_foot, self.gym, self.viewer, self.envs[i], sphere_pose)

        # return measured_heights, measured_foot_heights

    def normal_vector_to_quaternion(self, normal_vector):
        """법선 벡터를 쿼터니언으로 변환합니다."""
        # z축을 법선 벡터로 정렬하는 회전 행렬 생성
        z_axis = torch.tensor([0, 0, 1.0], device=normal_vector.device)
        # normal_vector의 차원에 맞게 z_axis 반복
        z_axis = z_axis.repeat(normal_vector.shape[0], 1)
        rotation_axis = torch.cross(z_axis, normal_vector)
        rotation_angle = torch.arccos(torch.sum(z_axis * normal_vector, dim=1) / (torch.norm(z_axis, dim=1) * torch.norm(normal_vector, dim=1)))

        # 회전 축이 정의되지 않은 경우(평행한 경우) 처리
        rotation_quaternion = torch.zeros((normal_vector.shape[0], 4), device=normal_vector.device)
        non_zero_axis = torch.linalg.norm(rotation_axis, dim=1) > 1e-6
        if torch.any(non_zero_axis):
            rotation_quaternion[non_zero_axis] = torch.tensor(Rotation.from_rotvec((rotation_axis[non_zero_axis] * rotation_angle[non_zero_axis].unsqueeze(1)).cpu().numpy()).as_quat(), device=normal_vector.device)
        rotation_quaternion[~non_zero_axis] = torch.tensor([0, 0, 0, 1.0], device=normal_vector.device) # 회전 없음

        return rotation_quaternion



# terrain generator
from isaacgym.terrain_utils import *

class Terrain:
    def __init__(self, cfg, num_robots) -> None:
        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20
        self.num_per_env = 2
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg["terrainProportions"]))]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain()   
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])
    
    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            

            terrain = SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.25, -0.15, 0, 0.15, 0.25]))
                    # random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.02, downsampled_scale=0.25)
                    random_uniform_terrain_with_flat_start(terrain, min_height=-0.05, max_height=0.05, step=0.02, downsampled_scale=0.25,flat_start_size=1.)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.25, -0.15, 0, 0.15, 0.25]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.03, 0.03])
                pyramid_stairs_terrain(terrain, step_width=0.1, step_height=step_height, platform_size=3.)
            elif choice < 1.:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length 
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                    width=self.width_per_env_pixels,
                                    length=self.width_per_env_pixels,
                                    vertical_scale=self.vertical_scale,
                                    horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                choice = j / num_terrains

                # slope = 0.025 + difficulty * 0.25
                # step_height = 0.025 + 0.025 * difficulty
                # discrete_obstacles_height = 0.025 + difficulty * 0.025
                # stepping_stones_size = 2 - 1.9* difficulty

                slope = 0.125 + difficulty * 0.25
                step_height = 0.01 + 0.15 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.1
                stepping_stones_size = 2 - 1.8* difficulty

                # slope = 0.025 + difficulty * 0.05
                # step_height = 0.025 + 0.05 * difficulty
                # discrete_obstacles_height = 0.025 + difficulty * 0.01
                # stepping_stones_size = 2 - 1.8* difficulty



                if choice < self.proportions[0]:
                    if choice < 0.1 :
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=1.5)
                elif choice < self.proportions[1]:
                    if choice < 0.3 :
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=1.5)
                    # random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.02, downsampled_scale=0.35)
                    random_uniform_terrain_with_flat_start(terrain, min_height=-0.1, max_height=0.1, step=0.02, downsampled_scale=0.4,flat_start_size=1.)
                elif choice < self.proportions[2]:
                    # random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.02, downsampled_scale=0.275)
                    random_uniform_terrain_with_flat_start(terrain, min_height=-0.1, max_height=0.1, step=0.02, downsampled_scale=0.4,flat_start_size=1.)
                elif choice < self.proportions[3]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 0.5, 1., 100, platform_size=3.)
                elif choice < self.proportions[4]:
                    if choice<0.88:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.4, step_height=step_height, platform_size=3.)
                    random_uniform_terrain_with_flat_start(terrain, min_height=-0.075, max_height=0.075, step=0.012, downsampled_scale=0.4,flat_start_size=3.)
                
                # else:
                #     stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0., platform_size=3.)

                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map +=1

                env_origin_x = (i + 0.5) * self.env_length 
                env_origin_y = (j + 0.5) * self.env_width 
                x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles