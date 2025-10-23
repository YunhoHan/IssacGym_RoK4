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

from isaacgym import gymtorch
from isaacgym import gymapi
from .base.vec_task import VecTask

import torch
from typing import Tuple, Dict

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, normalize, quat_apply, quat_rotate_inverse
from isaacgymenvs.tasks.base.vec_task import VecTask

# === 아래 ROS 관련 import 구문 추가 ===
import rospy
from geometry_msgs.msg import Twist # 로봇 속도 (선속도, 각속도)
from std_msgs.msg import Float32 # 개별 보상 값 등 단일 실수 값
from std_msgs.msg import Float32MultiArray # 관절 위치 배열 등 다중 실수 값
from sensor_msgs.msg import JointState # 관절 상태 (위치, 속도, 노력)
from sensor_msgs.msg import Joy # 조이스틱 입력

class RoK4Biped(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.C_BLACK = "\033[30m"
        self.C_RED = "\x1b[91m"
        self.C_GREEN = "\x1b[92m"
        self.C_YELLOW = "\x1b[93m"
        self.C_BLUE = "\x1b[94m"
        self.C_MAGENTA = "\x1b[95m"
        self.C_CYAN = "\x1b[96m"
        self.C_RESET = "\x1b[0m"

        self.cfg = cfg
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.init_done = False
        
        self.num_legs = 2

        self.min_swing_time = 0.35
        self.cycle_time = 1.5

        self.num_joint_actions = self.cfg["env"]["numJointActions"]

        # obs_scales normalization
        self.obs_scales = {}
        for obs_item, value in self.cfg["env"]["learn"]["obs_scales"].items():  # rewards 섹션 순회
            self.obs_scales[obs_item] = float(value)        
    
        # rew_scales normalization
        self.rew_scales = {}
        for rew_item, value in self.cfg["env"]["learn"]["rew_scales"].items():  # rewards 섹션 순회
            self.rew_scales[rew_item] = float(value)

        # pen_scales normalization
        self.pen_scales = {}
        for pen_item, value in self.cfg["env"]["learn"]["pen_scales"].items():  # rewards 섹션 순회
            self.pen_scales[pen_item] = float(value)

        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        #command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]

        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)

        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]

        self.Kp = self.cfg["env"]["control"]["Kp"]
        self.Kd = self.cfg["env"]["control"]["Kd"]

        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]

        for rew_item in self.rew_scales.keys():
            self.rew_scales[rew_item] *= self.dt

        for pen_item in self.pen_scales.keys():
            self.pen_scales[pen_item] *= self.dt 

        # === ROS 노드 초기화 및 퍼블리셔 생성 추가 ===
        # 관찰할 환경 ID 설정 (0번 환경 데이터만 시각화)
        self.observe_envs = 0
        
        try:
            rospy.init_node('rok4_plot_juggler_node', anonymous=True)
        except rospy.exceptions.ROSException:
            print("ROS node already initialized.") # 이미 초기화된 경우 에러 방지

        # 퍼블리셔 딕셔너리 생성 (보상 항목 등 동적 생성을 위해)
        self.ros_publishers = {}

        # 기본 상태 퍼블리셔 생성
        self.ros_publishers['command'] = rospy.Publisher('/rok4/command', Twist, queue_size=10)
        self.ros_publishers['base_velocity'] = rospy.Publisher('/rok4/base_velocity', Twist, queue_size=10)
        self.ros_publishers['joint_states'] = rospy.Publisher('/rok4/joint_states', JointState, queue_size=10)
        self.ros_publishers['actions'] = rospy.Publisher('/rok4/actions', JointState, queue_size=10) # 관절 액션만
        # self.ros_publishers['clock_actions'] = rospy.Publisher('/rok4/clock_actions', Twist, queue_size=10) # 시계 액션 (Twist 재활용)
        # self.ros_publishers['cycle_sin_cos'] = rospy.Publisher('/rok4/cycle_sin_cos', Twist, queue_size=10) # Sin/Cos (Twist 재활용)
        self.ros_publishers['total_reward'] = rospy.Publisher('/rok4/reward/total', Float32, queue_size=10)

        # 보상 항목별 퍼블리셔 동적 생성 (compute_reward 실행 후 채워짐)
        self.reward_container = {} # 각 보상 값을 임시 저장할 딕셔너리

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.Kp = torch.tensor(self.Kp, dtype=torch.float32, device=self.device)
        self.Kd = torch.tensor(self.Kd, dtype=torch.float32, device=self.device)
        print(self.C_CYAN,"Kp : ", self.Kp , self.C_RESET)
        print(self.C_CYAN,"Kd : ", self.Kd , self.C_RESET)
        print(self.C_CYAN,"Kp Size: ", self.Kp.size() , self.C_RESET)
        print(self.C_CYAN,"Kd Size: ", self.Kd.size() , self.C_RESET)

        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor)
        self.rigid_body_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,:3]
        self.rigid_body_vel = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,7:10]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel
        # self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.commands_scale = torch.tensor([self.obs_scales["linearVelocityScale"], self.obs_scales["linearVelocityScale"], self.obs_scales["angularVelocityScale"]], device=self.device, requires_grad=False,)

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # === 👇 게이트 클럭 관련 변수 추가 (rok3.py 참조) ===
        # Dof factors - Actions History & Clock
        # 액션 공간은 13개 관절 + 4개 클럭 = 17개. YAML 파일 수정 필요!
        # self.torques 텐서 크기 수정 (13개 관절 토크만 저장)

        # self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.torques = torch.zeros(self.num_envs, self.num_joint_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_actions = torch.zeros(self.num_envs, self.num_joint_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # 액션 이력 (Smoothness 보상용) - 관절 액션(13개)만 저장
        self.last_joint_actions = torch.zeros(self.num_envs, self.num_joint_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_joint_actions2 = torch.zeros(self.num_envs, self.num_joint_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # 클럭 액션 (정책 출력 중 4개) - YAML의 numActions(17) 중 나머지 4개
        self.clock_actions = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) #
        self.last_clock_actions = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) #

        # Dof factors - Dof History (Smoothness 보상용)
        self.last_dof_vel = torch.zeros_like(self.dof_vel) # 기존 라인 확인
        self.last_dof_vel2 = torch.zeros_like(self.dof_vel) #
        self.dof_acc = torch.zeros_like(self.dof_vel) #
        self.last_dof_acc = torch.zeros_like(self.dof_vel) #

        # Foot factors - Cycle (게이트 클럭 위상)
        self.sin_cycle = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device, requires_grad=False) #
        self.cos_cycle = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device, requires_grad=False) #
        self.cycle_t = torch.zeros(self.num_envs, self.num_legs, device=self.device, dtype=torch.float) # # L/R 시간
        self.cycle_L_x = torch.zeros(self.num_envs, device=self.device, dtype=torch.float) # # 로깅/보상용
        self.cycle_R_x = torch.zeros(self.num_envs, device=self.device, dtype=torch.float) # # 로깅/보상용

        self.phi = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device) # # 위상 오프셋

        # Foot factors - State (보상 함수용)
        self.last_foot_contacts = torch.zeros(self.num_envs, self.num_legs, device=self.device, dtype=torch.bool) #
        self.foot_pos = torch.zeros(self.num_envs, self.num_legs, 3, device=self.device, dtype=torch.float) # # 발 위치 저장용
        # self.feet_air_time -> self.foot_air_time 으로 이름 변경 및 크기 조정
        # self.feet_air_time = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_air_time = torch.zeros(self.num_envs, self.num_legs, device=self.device, dtype=torch.float) #->
        self.foot_swing_start_time = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device) #
        self.foot_swing_state = torch.zeros(self.num_envs, self.num_legs, dtype=torch.bool, device=self.device) #

        # === 👆 게이트 클럭 관련 변수 추가 완료 ===

        self.height_points = self.init_height_points()
        self.measured_heights = None
        # joint positions offsets
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_joint_actions):
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

    def _get_noise_scale_vec(self, cfg):
        # 이 벡터의 크기는 반드시 numObservations(48)와 일치해야 합니다.
        noise_vec = torch.zeros(self.cfg["env"]["numObservations"], device=self.device)
        # self.add_noise = self.cfg["env"]["learn"]["addNoise"]
        self.add_noise = self.cfg["env"]["learn"]["noise"]["addNoise"]
        # noise_level = self.cfg["env"]["learn"]["noiseLevel"]
        noise_level = self.cfg["env"]["learn"]["noise"]["noiseLevel"]

        # 슬라이스 인덱스는 compute_observations의 순서와 정확히 일치해야 합니다.
        # 0:3 -> 각속도
        noise_vec[0:3] = self.cfg["env"]["learn"]["noise"]["angularVelocityNoise"] * noise_level * self.obs_scales["angularVelocityScale"]
        # 3:6 -> 중력 벡터
        noise_vec[3:6] = self.cfg["env"]["learn"]["noise"]["gravityNoise"] * noise_level
        # 6:9 -> 커맨드 (노이즈 없음)
        noise_vec[6:9] = 0.
        # 9:22 -> 관절 각도 (13개)
        noise_vec[9:22] = self.cfg["env"]["learn"]["noise"]["dofPositionNoise"] * noise_level * self.obs_scales["dofPositionScale"]
        # 22:35 -> 관절 속도 (13개)
        noise_vec[22:35] = self.cfg["env"]["learn"]["noise"]["dofVelocityNoise"] * noise_level * self.obs_scales["dofVelocityScale"]
        # 35:48 -> 이전 행동 (13개, 노이즈 없음)
        noise_vec[35:48] = 0.
        
        # --- 추가된 관측값에 대한 노이즈 설정 (rok3.py 참조) ---
        # 48:50 -> sin_cycle (2개, 노이즈 없음)
        noise_vec[48:50] = 0.
        # 50:52 -> cos_cycle (2개, 노이즈 없음)
        noise_vec[50:52] = 0.
        # 52:56 -> clock_actions (4개, 노이즈 없음)
        noise_vec[52:56] = 0.
        # --- 👆 추가 완료 ---

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
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        asset_options.replace_cylinder_with_capsule = False
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        # prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(robot_asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)

        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        foot_name = self.cfg["env"]["urdfAsset"]["footName"]
        knee_name = self.cfg["env"]["urdfAsset"]["kneeName"]
        feet_names = [s for s in body_names if foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if knee_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(robot_asset)

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.robot_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_prop)
            robot_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            self.envs.append(env_handle)
            self.robot_handles.append(robot_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], "base")

    def check_termination(self):
        self.reset_buf = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        if not self.allow_knee_contacts:
            knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
            self.reset_buf |= torch.any(knee_contact, dim=1)

        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_observations(self):
        # === 👇 함수 전체 (56개 관측값 생성) ===

        # --- 게이트 클럭 위상 계산 (post_physics_step에서 cycle_t 업데이트 후 실행됨) ---
        # TODO: post_physics_step에서 self.cycle_t 업데이트 로직 추가 필요
        # TODO: pre_physics_step에서 처리된 self.clock_actions[:, :2]가 self.phi 값으로 사용될 수 있음

        # 가정: self.cycle_t 는 (num_envs, 2) 크기, 0~1 사이 값
        # 가정: self.clock_actions[:, :2] 는 각 다리의 위상 오프셋 phi 값 (스무딩 후) L/R 다리 위상을 pi 만큼 차이 나게 계산

        phi_offsets = self.clock_actions[:, :2] * ~self.no_commands.unsqueeze(1) #

        # 각 다리의 위상 계산 (rok3.py 방식 참고)
        # cycle_t는 L/R 다리 각각의 시간 진행률 (0~1)
        cycle_rad_L = 2. * torch.pi * (self.cycle_t[:, 0] / self.cycle_time) + phi_offsets[:, 0]
        cycle_rad_R = 2. * torch.pi * (self.cycle_t[:, 1] / self.cycle_time) + phi_offsets[:, 1] + torch.pi # rok3 방식은 R에 pi 추가

        self.sin_cycle[:, 0] = torch.sin(cycle_rad_L)
        self.cos_cycle[:, 0] = torch.cos(cycle_rad_L)
        
        self.sin_cycle[:, 1] = torch.sin(cycle_rad_R)
        self.cos_cycle[:, 1] = torch.cos(cycle_rad_R)
        # --- 게이트 위상 계산 완료 ---

        # 스케일링된 관측값 준비
        base_ang_vel_scaled = self.base_ang_vel * self.obs_scales["angularVelocityScale"]
        dof_pos_scaled = self.dof_pos * self.obs_scales["dofPositionScale"]
        dof_vel_scaled = self.dof_vel * self.obs_scales["dofVelocityScale"]

        # 56개 관측값 합치기 (rok3.py 순서 참조)
        self.obs_buf = torch.cat((  base_ang_vel_scaled,         # 3   [0:3]
                                    self.projected_gravity,      # 3   [3:6]
                                    self.commands[:, :3],        # 3   [6:9]
                                    dof_pos_scaled,              # 13  [9:22]
                                    dof_vel_scaled,              # 13  [22:35]
                                    self.joint_actions,          # 13  [35:48] # 앞선 pre에서 계산된 actions 사용
                                    self.sin_cycle,              # 2   [48:50] <-- 추가
                                    self.cos_cycle,              # 2   [50:52] <-- 추가
                                    self.clock_actions           # 4   [52:56] <-- 추가
                                    ), dim=-1)
        # === 👆 함수 전체 수정 완료 ===
        
        # # 총 차원: 3(각속도) + 3(중력) + 3(커맨드) + 13(관절각도) + 13(관절속도) + 13(이전행동) = 48
        # self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales["angularVelocityScale"],
        #                             self.projected_gravity,
        #                             self.commands[:, :3], # 우선 3개의 커맨드만 사용
        #                             self.dof_pos * self.obs_scales["dofPositionScale"],
        #                             self.dof_vel * self.obs_scales["dofVelocityScale"],
        #                             self.joint_actions
        #                             ), dim=-1)
        
    # === plot_juggler 함수 추가 ===
    def plot_juggler(self):
        # 관찰할 환경의 데이터만 추출
        env_id = self.observe_envs

        # --- 커맨드 ---
        cmd_msg = Twist()
        cmd_msg.linear.x = self.commands[env_id, 0].item()
        cmd_msg.linear.y = self.commands[env_id, 1].item()
        cmd_msg.angular.z = self.commands[env_id, 2].item()
        if 'command' in self.ros_publishers:
            self.ros_publishers['command'].publish(cmd_msg)

        # --- 로봇 베이스 속도 ---
        vel_msg = Twist()
        vel_msg.linear.x = self.base_lin_vel[env_id, 0].item()
        vel_msg.linear.y = self.base_lin_vel[env_id, 1].item()
        vel_msg.linear.z = self.base_lin_vel[env_id, 2].item()
        vel_msg.angular.x = self.base_ang_vel[env_id, 0].item()
        vel_msg.angular.y = self.base_ang_vel[env_id, 1].item()
        vel_msg.angular.z = self.base_ang_vel[env_id, 2].item()
        if 'base_velocity' in self.ros_publishers:
            self.ros_publishers['base_velocity'].publish(vel_msg)

        # --- 관절 상태 ---
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = self.dof_names[:self.num_actions] # 현재 num_actions (13) 사용
        joint_msg.position = self.dof_pos[env_id, :self.num_actions].cpu().numpy()
        joint_msg.velocity = self.dof_vel[env_id, :self.num_actions].cpu().numpy()
        joint_msg.effort = self.torques[env_id, :self.num_actions].cpu().numpy()
        if 'joint_states' in self.ros_publishers:
            self.ros_publishers['joint_states'].publish(joint_msg)

        # --- 액션 (관절 부분) ---
        action_msg = JointState()
        action_msg.header.stamp = rospy.Time.now()
        action_msg.name = [f"action_{name}" for name in self.dof_names[:self.num_actions]]
        action_msg.position = self.joint_actions[env_id, :self.num_actions].cpu().numpy()
        if 'actions' in self.ros_publishers:
            self.ros_publishers['actions'].publish(action_msg)
            
        # --- 개별 보상 항목 ---
        for reward_name, reward_value in self.reward_container.items():
            topic_name = f'/rok4/reward/{reward_name}'
            if topic_name not in self.ros_publishers:
                self.ros_publishers[topic_name] = rospy.Publisher(topic_name, Float32, queue_size=10)
            reward_msg = Float32()
            reward_msg.data = reward_value.item() # / self.dt # 필요시 dt로 나누어 원래 스케일 확인
            self.ros_publishers[topic_name].publish(reward_msg)

        # --- 총 보상 ---
        total_reward_msg = Float32()
        total_reward_msg.data = self.rew_buf[env_id].item() # / self.dt
        if 'total_reward' in self.ros_publishers:
            self.ros_publishers['total_reward'].publish(total_reward_msg)
    # =================================================

    def compute_reward(self):
        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])

        rew_lin_vel_xy = torch.exp(-lin_vel_error/0.1) * self.rew_scales["linearVelocityXYRewardScale"] #
        rew_ang_vel_z = torch.exp(-ang_vel_error/0.1) * self.rew_scales["angularVelocityZRewardScale"] #

        # other base velocity penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.pen_scales["linearVelocityZRewardScale"] #
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.pen_scales["angularVelocityXYRewardScale"] #

        # orientation penalty
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.pen_scales["orientationRewardScale"] #

        # base height penalty
        rew_base_height = torch.square(self.root_states[:, 2] - self.cfg["env"]["baseInitState"]["pos"][2]) * self.pen_scales["baseHeightRewardScale"] # # TODO target base height cfg로 이동

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.pen_scales["torqueRewardScale"] #

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.pen_scales["jointAccRewardScale"] #

        # collision penalty
        knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
        rew_collision = torch.sum(knee_contact, dim=1) * self.pen_scales["kneeCollisionRewardScale"] #

        # stumbling penalty
        stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5.) * (torch.abs(self.contact_forces[:, self.feet_indices, 2]) < 1.)
        rew_stumble = torch.sum(stumble, dim=1) * self.pen_scales["feetStumbleRewardScale"] #

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_joint_actions - self.joint_actions), dim=1) * self.pen_scales["actionRateRewardScale"] #

        self.foot_pos = self.rigid_body_pos[:, self.feet_indices, :]
        foot_velocities = self.rigid_body_vel[:, self.feet_indices, :]
        foot_contact_forces = self.contact_forces[:, self.feet_indices, :]
        self.foot_contact = torch.norm(foot_contact_forces, dim=-1) > 1.

        # air time reward
        # contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 1.
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        first_contact = (self.foot_air_time > 0.) * contact
        self.foot_air_time += self.dt
        rew_airTime = torch.sum((self.foot_air_time - 0.5) * first_contact, dim=1) * self.rew_scales["feetAirTimeRewardScale"] #
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.foot_air_time *= ~contact

        # cosmetic penalty for hip motion
        # RoK-4의 Hip Roll 관절 인덱스인 1번과 7번으로 수정합니다.
        rew_hip = torch.sum(torch.abs(self.dof_pos[:, [1, 7]] - self.default_dof_pos[:, [1, 7]]), dim=1)* self.pen_scales["hipRewardScale"] #
   
        # Swing/Stance 일관성 패널티
 
        # 스윙 비율 및 시작 시점 정의 (하이퍼파라미터)
        swing_ratio = 0.4  # 예: 전체 주기의 40%를 스윙에 할당
        stance_ratio = 1.0 - swing_ratio  # 예: 전체 주기의 60%를 스탠스에 할당
        swing_start_phase = 0.5 + (stance_ratio-swing_ratio)/4 # 예: 스윙 시작 시점 (0.5에서 약간 오프셋)

        # 현재 시간을 0~1 사이의 "정규화된 위상(phase)"으로 변환
        normalized_phase = self.cycle_t / self.cycle_time # (값은 0~1 사이)

        # 스윙 종료 시점 계산 (1.0이 넘어가면 래핑(wrapping) 처리)
        swing_end_phase = (swing_start_phase + swing_ratio) % 1.0

        # 위상 정의 (경계 래핑(wrapping) 처리)
        if swing_start_phase < swing_end_phase:
            # Case 1: 래핑 없음 (예: start=0.5, end=0.9)
            swing_phase = (normalized_phase >= swing_start_phase) & \
                          (normalized_phase < swing_end_phase)
        else:
            # Case 2: 래핑 발생 (예: start=0.8, end=0.2)
            swing_phase = (normalized_phase >= swing_start_phase) | \
                          (normalized_phase < swing_end_phase)
        
        # 스탠스 구간은 스윙 구간의 반대
        stance_phase = ~swing_phase

        
        foot_force_norm = torch.norm(foot_contact_forces, dim=-1)
        foot_velocities_squared_sum = torch.sum(torch.square(foot_velocities), dim=-1)

        swing_stance_penalty = torch.clip(torch.sum(foot_velocities_squared_sum * stance_phase, dim=-1), 0, 10) * 0.05 + \
                               torch.clip(torch.sum(foot_force_norm * swing_phase, dim=-1), 0, 30) * 0.025
        
        pen_swing_stance_phase = swing_stance_penalty * ~self.no_commands * self.pen_scales["swingStancePhase"]

        # 최소 스윙 시간 패널티
        self.foot_swing_state[~self.foot_contact] = True
        self.foot_swing_state[self.foot_contact] = False
        
        current_time_tensor = self.progress_buf * self.dt # 현재 시간 (에피소드 기준)
        self.foot_swing_start_time[~self.foot_contact] = current_time_tensor.unsqueeze(1).repeat(1, self.num_legs)[~self.foot_contact]

        swing_time = current_time_tensor.unsqueeze(1) - self.foot_swing_start_time
        contact_made = (swing_time > 0.) * self.foot_contact # 스윙 중(시간>0)에 착지(contact)
        
        pen_short_swing = torch.sum(torch.relu(self.min_swing_time - swing_time) * contact_made, dim=-1) * self.pen_scales["shortSwing"]

        # 3. 발 높이(Foot Clearance) 패널티
        ref_foot_height = 0.075
        # 스윙 단계(swing_phase)이고, 커맨드가 있을 때(~self.no_commands)만 패널티 적용
        foot_height_err = torch.sum(torch.relu(ref_foot_height - self.foot_pos[:, :, 2]), dim=-1)
        pen_foot_height = foot_height_err * torch.any(swing_phase, dim=-1) * ~self.no_commands * self.pen_scales["footHeight"]
        
        # 4. 정지 시 기본 자세 패널티
        default_pos_err = torch.square(self.default_dof_pos - self.dof_pos)
        pen_default_pos_standing = torch.sum(default_pos_err, dim=-1) * self.no_commands * self.pen_scales["defaultPosStanding"]

        # === 개별 보상 값 저장을 위한 코드 추가 ===
        self.reward_container.clear()
        reward_terms = [
             "rew_lin_vel_xy", "rew_ang_vel_z", "rew_lin_vel_z", "rew_ang_vel_xy",
             "rew_orient", "rew_base_height", "rew_torque", "rew_joint_acc",
             "rew_collision", "rew_action_rate", "rew_airTime",
             "rew_hip", "rew_stumble", "pen_swing_stance_phase", "pen_short_swing", "pen_foot_height", "pen_default_pos_standing"
        ]

        for term in reward_terms:
             if term in locals():
                 self.reward_container[term] = locals()[term][self.observe_envs]

        # total reward
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height +\
                       rew_torque + rew_joint_acc + rew_collision + rew_action_rate + rew_airTime + rew_hip + rew_stumble +\
                    pen_swing_stance_phase + pen_short_swing + pen_foot_height + pen_default_pos_standing
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # add termination reward
        self.rew_buf += self.rew_scales["terminalReward"] * self.reset_buf * ~self.timeout_buf #

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["collision"] += rew_collision
        self.episode_sums["stumble"] += rew_stumble
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["air_time"] += rew_airTime
        self.episode_sums["base_height"] += rew_base_height
        self.episode_sums["hip"] += rew_hip

    def reset_idx(self, env_ids):
        positions_offset = torch_rand_float(0.9, 1.1, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1) # set small commands to zero

        self.last_joint_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.foot_air_time[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self.cycle_t[env_ids,:] = 0.
        self.last_clock_actions[env_ids] = 0.

        self.last_foot_contacts[env_ids] = 0.
        self.foot_swing_start_time[env_ids] = 0.
        self.foot_swing_state[env_ids] = False

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2])*self.max_episode_length_s*0.25)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def push_robots(self):
        self.root_states[:, 7:9] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def pre_physics_step(self, actions):
        # actions 텐서의 크기는 (num_envs, 17)

        # === 👇 액션 분리 로직  ===
        # 17개 액션을 13개(관절)와 4개(클럭)으로 분리
        joint_actions_raw = actions[:, :self.num_joint_actions] # 0 ~ 12번 인덱스 (13개)
        self.clock_actions = actions[:, self.num_joint_actions:].clone().to(self.device) # 13 ~ 16번 인덱스 (4개)

        # 관절 액션에 action_scale 적용
        # self.joint_actions 변수에 최종 관절 액션 (13개) 저장
        self.joint_actions = joint_actions_raw * self.action_scale # 크기: (num_envs, 13)

        # === 👇 클럭 액션 처리 로직  ===
        # TODO: 아래 게인 값들은 하이퍼파라미터이므로 나중에 조정이 필요할 수 있습니다.
        phi_gain = 0.5   # 위상 오프셋 스무딩 게인
        delta_gain = 0.5 # 주기 변화 스무딩 게인

        # 클럭 액션 스케일링 (앞 2개는 위상 오프셋, 뒤 2개는 주기 변화량)
        self.clock_actions[:, :2]  *= 1.0  # 위상 오프셋 (phi) 스케일
        self.clock_actions[:, 2:4] *= 0.025 # 주기 변화량 (delta) 스케일
        self.clock_actions[:, 2:4] = torch.abs(self.clock_actions[:, 2:4]) # 주기 변화량은 양수로

        # 클럭 액션 스무딩 (EMA 필터와 유사)
        self.clock_actions[:, :2]  = self.last_clock_actions[:, :2] * (1 - phi_gain) + self.clock_actions[:, :2] * phi_gain
        self.clock_actions[:, 2:4] = self.last_clock_actions[:, 2:4] * (1 - delta_gain) + self.clock_actions[:, 2:4] * delta_gain
        # === 👆 클럭 액션 처리 완료 ===

        # === 👇 PD 제어 루프 (기존 코드와 유사하나 self.joint_actions 사용) ===
        for i in range(self.decimation):
            # PD 타겟 계산 시 action_scale이 이미 적용된 self.joint_actions (13개) 사용
            targets = self.joint_actions + self.default_dof_pos # 크기: (num_envs, 13)

            # 토크 계산 (모든 텐서 크기가 13으로 일치)
            torques = self.Kp * (targets - self.dof_pos) - self.Kd * self.dof_vel # 크기: (num_envs, 13)

            # TODO: 토크 제한 값 (-80, 80)을 YAML 파일에서 읽어오도록 수정 필요 (예: self.torque_limits)
            torques = torch.clip(torques, -300., 300.)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques = torques.view(self.torques.shape) # 계산된 토크 저장 (13개)

            # 시뮬레이션 및 상태 업데이트
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # self.joint_actions = actions.clone().to(self.device)
        # for i in range(self.decimation):
        #     torques = torch.clip(self.Kp*(self.action_scale*self.joint_actions + self.default_dof_pos - self.dof_pos) - self.Kd*self.dof_vel,
        #                          -80., 80.)
        #     self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        #     self.torques = torques.view(self.torques.shape)
        #     self.gym.simulate(self.sim)
        #     if self.device == 'cpu':
        #         self.gym.fetch_results(self.sim, True)
        #     self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):
        # self.gym.refresh_dof_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        if self.common_step_counter % self.push_interval == 0:
            self.push_robots()

        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # forward = quat_apply(self.base_quat, self.forward_vec)
        # heading = torch.atan2(forward[:, 1], forward[:, 0])
        # self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # check for no commands 
        self.no_commands = (torch.norm(self.commands,dim=-1) == 0)

        # update cycle time
        self.cycle_t[:,0] += self.clock_actions[:,2]
        self.cycle_t[:,1] += self.clock_actions[:,3]
        self.cycle_t *= ~self.no_commands.unsqueeze(1)
        self.cycle_t = torch.where(self.cycle_t > self.cycle_time, 
                                   self.cycle_t - self.cycle_time, 
                                   self.cycle_t)
        
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.plot_juggler()
        
        self.last_joint_actions[:] = self.joint_actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # draw height lines
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False) # 10-50cm on each side
        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False) # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self, env_ids=None):
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
 
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale


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
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
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

                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice<self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0., platform_size=3.)

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
