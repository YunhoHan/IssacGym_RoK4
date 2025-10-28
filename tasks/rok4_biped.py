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

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, normalize, quat_apply, quat_rotate_inverse, get_euler_xyz, my_quat_rotate

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("Scipy not found. Please install it: pip install scipy")
    R = None

from isaacgymenvs.tasks.base.vec_task import VecTask

# === 아래 ROS 관련 import 구문 추가 ===
import rospy
from geometry_msgs.msg import Twist # 로봇 속도 (선속도, 각속도)
from std_msgs.msg import Float32 # 개별 보상 값 등 단일 실수 값
from std_msgs.msg import Float32MultiArray # 관절 위치 배열 등 다중 실수 값
from geometry_msgs.msg import Point # 3차원 점
from geometry_msgs.msg import Quaternion, Vector3 
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
        self.test_mode = self.cfg["env"]["test"]
        self.init_done = False
        self.need_reset = False

        self.num_legs = 2

        self.min_swing_time = 0.35
        self.cycle_time = 1.2

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

        ranges_cfg = self.cfg["env"]["EnvsNumRanges"]
        current_end = -1

        self.stand_env_range = [current_end + 1, current_end + ranges_cfg["stand_env_range"]]
        current_end += ranges_cfg["stand_env_range"]
        self.only_plus_x_envs_range = [current_end + 1, current_end + ranges_cfg["only_plus_x_envs_range"]]
        current_end += ranges_cfg["only_plus_x_envs_range"]
        self.only_minus_x_envs_range = [current_end + 1, current_end + ranges_cfg["only_minus_x_envs_range"]]
        current_end += ranges_cfg["only_minus_x_envs_range"]
        self.only_plus_y_envs_range = [current_end + 1, current_end + ranges_cfg["only_plus_y_envs_range"]]
        current_end += ranges_cfg["only_plus_y_envs_range"]
        self.only_minus_y_envs_range = [current_end + 1, current_end + ranges_cfg["only_minus_y_envs_range"]]
        current_end += ranges_cfg["only_minus_y_envs_range"]
        self.only_plus_yaw_envs_range = [current_end + 1, current_end + ranges_cfg["only_plus_yaw_envs_range"]]
        current_end += ranges_cfg["only_plus_yaw_envs_range"]
        self.only_minus_yaw_envs_range = [current_end + 1, current_end + ranges_cfg["only_minus_yaw_envs_range"]]
        current_end += ranges_cfg["only_minus_yaw_envs_range"]
        self.plus_x_plus_yaw_envs_range = [current_end + 1, current_end + ranges_cfg["plus_x_plus_yaw_envs_range"]]
        current_end += ranges_cfg["plus_x_plus_yaw_envs_range"]
        self.plus_x_minus_yaw_envs_range = [current_end + 1, current_end + ranges_cfg["plus_x_minus_yaw_envs_range"]]
        current_end += ranges_cfg["plus_x_minus_yaw_envs_range"]
        self.minus_x_plus_yaw_envs_range = [current_end + 1, current_end + ranges_cfg["minus_x_plus_yaw_envs_range"]]
        current_end += ranges_cfg["minus_x_plus_yaw_envs_range"]
        self.minus_x_minus_yaw_envs_range = [current_end + 1, current_end + ranges_cfg["minus_x_minus_yaw_envs_range"]]
        current_end += ranges_cfg["minus_x_minus_yaw_envs_range"]

        # 계산된 범위 출력 (확인용)
        print(self.C_CYAN + "Calculated Environment Ranges:" + self.C_RESET)
        print(f"  Stand      : {self.stand_env_range}")
        print(f"  +X Only    : {self.only_plus_x_envs_range}")
        print(f"  -X Only    : {self.only_minus_x_envs_range}")
        print(f"  +Y Only    : {self.only_plus_y_envs_range}")
        print(f"  -Y Only    : {self.only_minus_y_envs_range}")
        print(f"  +Yaw Only  : {self.only_plus_yaw_envs_range}")
        print(f"  -Yaw Only  : {self.only_minus_yaw_envs_range}")
        print(f"  +X +Yaw    : {self.plus_x_plus_yaw_envs_range}")
        print(f"  +X -Yaw    : {self.plus_x_minus_yaw_envs_range}")
        print(f"  -X +Yaw    : {self.minus_x_plus_yaw_envs_range}")
        print(f"  -X -Yaw    : {self.minus_x_minus_yaw_envs_range}")

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
        self.policy_dt = (self.decimation + 1) * self.cfg["sim"]["dt"]

        print(self.C_CYAN,"Simulation dt: ", self.policy_dt , self.C_RESET)

        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"] 
        self.max_episode_length = int(self.max_episode_length_s/ self.policy_dt + 0.5)

        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.policy_dt + 0.5)

        self.freeze_cnt = 0 # freeze 상태 지속 시간 카운터
        self.freeze_flag = False # 현재 freeze 상태인지 여부
        self.freeze_steps = 50 # freeze 상태를 유지할 스텝 수 (나중에 랜덤화 가능)
        # YAML에서 freeze 간격 읽어오기 (없으면 기본값 설정)
        freeze_interval_s = self.cfg["env"]["learn"].get("freezeInterval_s", 15.0) # 기본 15초
        self.freeze_interval = int(freeze_interval_s / self.policy_dt + 0.5)

        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]

        self.Kp = self.cfg["env"]["control"]["Kp"]
        self.Kd = self.cfg["env"]["control"]["Kd"]

        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]

        for rew_item in self.rew_scales.keys():
            self.rew_scales[rew_item] *= self.policy_dt

        for pen_item in self.pen_scales.keys():
            self.pen_scales[pen_item] *= self.policy_dt 

        if self.test_mode:
            self.observe_envs = 0
            self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
            self.cam_change_flag = True
            self.cam_change_cnt = 0
            print('\033[103m' + '\033[33m' + '_________________________________________________________' + '\033[0m')
            print('\033[103m' + '\033[33m' + '________________________' + '\033[30m' + 'Test Mode' + '\033[33m' + '\033[103m'+ '________________________' + '\033[0m')
            print('\033[103m' + '\033[33m' +'_________________________________________________________' + '\033[0m')
        else:
            # 관찰할 환경 ID 설정 (0번 환경 데이터만 시각화)
            self.observe_envs = 200
        
        # === ROS Plot Juggler 노드 초기화 및 퍼블리셔 생성 추가 ===
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
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
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

        # --- ↓↓↓ CoM 계산 관련 변수 추가 (rok3.py 참고) ---
        self.robot_CoM = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.body_CoM = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float32, device=self.device)
        self.total_mass = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.body_mass = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.float32, device=self.device)

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
        self.episode_sums = {"rew_lin_vel_xy": torch_zeros(), "rew_ang_vel_z": torch_zeros(), "rew_feet_air_time": torch_zeros(), "rew_hip": torch_zeros(),
                            "pen_lin_vel_z": torch_zeros(), "pen_ang_vel_xy": torch_zeros(), "pen_baseOrientation": torch_zeros(), "pen_baseHeight": torch_zeros(),
                            "pen_swing_stance_phase": torch_zeros(), "pen_foot_slip": torch_zeros(), "pen_foot_height": torch_zeros(),
                            "pen_footGapY": torch_zeros(), "pen_legGapY": torch_zeros(), "pen_footGapXStanding": torch_zeros(),
                            "pen_hipYaw": torch_zeros(), "pen_hipRoll": torch_zeros(), "pen_ankleRoll": torch_zeros(),
                            "pen_Knee_collision": torch_zeros(), "pen_Calf_collision": torch_zeros(), "pen_jump": torch_zeros(),
                            "pen_dofLimit": torch_zeros(), "pen_actionLimit": torch_zeros(), "pen_dofVelLimit": torch_zeros(),
                            "pen_torque": torch_zeros(), "pen_joint_acc": torch_zeros(), "pen_action_rate": torch_zeros(), "pen_actionSmoothness2": torch_zeros(), "pen_jointEnergy": torch_zeros(),
                            "pen_default_pos_standing": torch_zeros(),
                            }
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
        if self.test_mode:
            asset_file = self.cfg["env"]["urdfAsset"]["test_file"]

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
        calf_name = self.cfg["env"]["urdfAsset"]["calfName"]
        feet_names = [s for s in body_names if foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if knee_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        calf_names = [s for s in body_names if calf_name in s]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)

        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        # --- ↓↓↓ 관절 한계 변수 추가 (rok3.py 참고) ---
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        base_friction = self.cfg["env"]["dof_properties"]["friction"]
        base_damping = self.cfg["env"]["dof_properties"]["damping"]
        base_armature = self.cfg["env"]["dof_properties"]["armature"]

        # --- ↓↓↓ [ 3단계 ] 관절 한계 초기화 및 로봇 설정 버퍼 채우기 (rok3.py 참고) ---
        for i in range(self.num_dof):
            # 관절 위치 한계 저장
            self.dof_pos_limits[i, 0] = dof_props["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props["upper"][i].item()
            # 관절 속도 한계 저장
            self.dof_vel_limits[i] = dof_props["velocity"][i].item()
            # (선택적) Isaac Gym 내부 PD 제어기 비활성화 (우리 코드에서 직접 PD 제어하므로)
            dof_props['stiffness'][i] = 0

            dof_props['friction'][i] = base_friction[i] * 0.0
            dof_props['damping'][i] = base_damping[i] * 0.0 # <- 여기가 수정됨!
            dof_props['armature'][i] = base_armature[i] * 0.0
        # --- ↑↑↑ ---

        self.joint_torque_limits = torch.tensor(self.cfg["env"]["dof_properties"]["torque_limits"], dtype=torch.float32, device=self.device)

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

            # --- ↓↓↓ 로봇 설정 버퍼 채우기 (CoM 계산용) ---
            actor_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)

            # 로봇 설정 버퍼 (CoM 계산 시 각 바디의 로컬 CoM 위치와 질량 저장용)
            self.robot_config_buffer = torch.empty(self.num_envs, self.num_bodies, 4, dtype=torch.float32, device=self.device)

            for j in range(self.num_bodies):
                self.robot_config_buffer[i, j, 0] = actor_rigid_body_props[j].com.x
                self.robot_config_buffer[i, j, 1] = actor_rigid_body_props[j].com.y
                self.robot_config_buffer[i, j, 2] = actor_rigid_body_props[j].com.z
                self.robot_config_buffer[i, j, 3] = actor_rigid_body_props[j].mass
            # --- ↑↑↑ ---

            self.envs.append(env_handle)
            self.robot_handles.append(robot_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], knee_names[i])
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], calf_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], "base")

    def check_termination(self):
        self.reset_buf = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        if not self.allow_knee_contacts:
            knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
            self.reset_buf |= torch.any(knee_contact, dim=1)

        if self.test_mode:
            need_reset_reset = self.need_reset
            self.reset_buf = self.reset_buf | need_reset_reset

        # 발보다 베이스가 낮으면 넘어짐 (Fall detection 1: base lower than feet)
        self.fall = torch.any(self.rigid_body_pos[:, 0, 2].unsqueeze(1) < self.rigid_body_pos[:, self.feet_indices, 2], dim=-1)
        self.reset_buf |= self.fall

        # 2. 베이스 높이가 너무 낮거나 높으면 넘어짐 (Fall detection 2: base height limits)
        self.fall2 = (self.rigid_body_pos[:, 0, 2] < 0.3) | (self.rigid_body_pos[:, 0, 2] > 1.5) # 높이 임계값은 조정 가능
        self.reset_buf |= self.fall2

        # 시간 초과에 따른 리셋
        if not (self.test_mode):
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
        
    def compute_reward(self):
        # === 필요 변수 계산 ===
        # 발 위치, 속도, 접촉 상태 (기존 코드)
        self.foot_pos = self.rigid_body_pos[:, self.feet_indices, :]
        foot_velocities = self.rigid_body_vel[:, self.feet_indices, :]
        foot_contact_forces = self.contact_forces[:, self.feet_indices, :]
        self.foot_contact = torch.norm(foot_contact_forces, dim=-1) > 1.

        # # === CoM 계산 ===
        # # (주의: 랜덤화가 비활성화된 경우 body_mass_noise는 0임)
        # # self.body_mass = self.robot_config_buffer[:, :, 3] # + self.body_mass_noise (랜덤화 시 추가)
        # self.body_mass = self.robot_config_buffer[:, :, 3] # 우선 랜덤화 없다고 가정
        # self.total_mass = torch.sum(self.body_mass, dim=-1)
        # # 각 바디의 월드 좌표계 CoM 위치 계산 (회전 적용)
        # # (my_quat_rotate 함수 필요 - isaacgymenvs.utils.torch_jit_utils 에서 가져오거나 직접 구현)
        # # body_CoM_local = self.robot_config_buffer[:, :, :3]
        # # body_CoM_rotated = quat_rotate(self.base_quat.unsqueeze(1).repeat(1, self.num_bodies, 1), body_CoM_local) # shape 확인 필요
        # # self.body_CoM = self.rigid_body_pos + body_CoM_rotated # Base 기준 -> 월드 좌표
        # # => 복잡하므로 우선 Base 위치로 근사하거나, 나중에 정확한 계산 추가
        # # 임시: Base 위치를 CoM 위치로 사용
        # com_pos = self.root_states[:, :3]
        # # 임시: Base 속도를 CoM 속도로 사용
        # com_vel = self.base_lin_vel # Z축 속도만 필요할 수 있음

        # === Base Motion Rewards & Penalties ===
        # lin_vel_error_xy = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error_xy = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=-1)
        ang_vel_error_z = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error_xy / 0.2) * self.rew_scales["linearVelocityXYRewardScale"]
        rew_ang_vel_z = torch.exp(-ang_vel_error_z / 0.05) * self.rew_scales["angularVelocityZRewardScale"]

        # Undesired Velocities (원하는 움직임 이외의 속도에 대한 패널티)
        pen_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.pen_scales["linearVelocityZRewardScale"]
        pen_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.pen_scales["angularVelocityXYRewardScale"]

        # Body Orientation Penalty ===
        base_ori_err = torch.norm(self.projected_gravity[:, :2], dim=-1)
        pen_baseOrientation = base_ori_err * self.pen_scales["baseOrientation"]

        # Base Height Penalty ===
        target_base_height = 0.905
        base_height_err = torch.abs(self.root_states[:, 2] - target_base_height)
        pen_baseHeight = base_height_err * self.pen_scales["baseHeightRewardScale"]

        # # === CoM Stability Penalties (안정성) ===
        # # CoM Height Penalty (목표 높이 유지)
        # target_com_height = self.cfg["env"]["baseInitState"]["pos"][2]
        # com_height_err = torch.abs(com_pos[:, 2] - target_com_height)
        # pen_comHeight = com_height_err * self.pen_scales["comHeight"]

        # # CoM Support Penalty (지지 다각형 내 유지) - 단순화된 버전 (두 발 사이 중간점 기준)
        # # (더 정확하려면 실제 지지 다각형 계산 필요)
        # foot_pos_xy = self.foot_pos[:, :, :2]
        # com_pos_xy = com_pos[:, :2]
        # # 두 발이 땅에 닿았을 때만 계산 (혹은 한 발일 때 그 발 위치 기준)
        # both_feet_on_ground = torch.sum(self.foot_contact, dim=-1) == 2
        # single_foot_on_ground = torch.sum(self.foot_contact, dim=-1) == 1
        # support_center = torch.zeros_like(com_pos_xy)
        # # 두 발 지지: 두 발의 중간 지점
        # support_center[both_feet_on_ground] = torch.mean(foot_pos_xy[both_feet_on_ground], dim=1)
        # # 한 발 지지: 땅에 닿은 발 위치 (L=0, R=1)
        # left_support = single_foot_on_ground & self.foot_contact[:, 0]
        # right_support = single_foot_on_ground & self.foot_contact[:, 1]
        # support_center[left_support] = foot_pos_xy[left_support, 0, :]
        # support_center[right_support] = foot_pos_xy[right_support, 1, :]
        # # CoM과 지지 중심 사이의 거리 계산 (발이 땅에 있을 때만 패널티)
        # com_support_err = torch.norm(com_pos_xy - support_center, dim=-1)
        # pen_comSupport = com_support_err * (num_contact_feet > 0) * self.pen_scales["comSupport"]

        # # CoM Vertical Velocity Penalty
        # pen_comVelZ = torch.square(com_vel[:, 2]) * self.pen_scales["comVelZ"]

        # === Gait Related Rewards & Penalties ===
        # Gait Consistency (S-S-S 비율 기반)
        swing_ratio = 0.4
        stance_ratio = 1.0 - swing_ratio
        swing_start_phase = 0.5 + (stance_ratio - swing_ratio) / 4
        normalized_phase = self.cycle_t / self.cycle_time
        swing_end_phase = (swing_start_phase + swing_ratio) % 1.0
        if swing_start_phase < swing_end_phase:
            swing_phase = (normalized_phase >= swing_start_phase) & (normalized_phase < swing_end_phase)
        else:
            swing_phase = (normalized_phase >= swing_start_phase) | (normalized_phase < swing_end_phase)
        stance_phase = ~swing_phase

        # 발에 가해지는 접촉 힘 벡터의 크기(norm)를 계산합니다. (힘의 총량)
        foot_force_norm = torch.norm(foot_contact_forces, dim=-1)
        # 발 속도 벡터의 각 성분(x, y, z)을 제곱하고 합하여, 발이 얼마나 빠르게 움직이는지를 나타내는 스칼라 값을 계산합니다.
        foot_velocities_squared_sum = torch.sum(torch.square(foot_velocities), dim=-1)
        # 스탠스 패널티(미끄러짐) + 스윙 패널티(접촉)를 합산합니다.
        swing_stance_penalty = torch.clip(torch.sum(foot_velocities_squared_sum * stance_phase, dim=-1), 0, 10) * 0.05 + \
                               torch.clip(torch.sum(foot_force_norm * swing_phase, dim=-1), 0, 30) * 0.025
        # 최종 패널티 계산: 위에서 계산된 패널티에 YAML 스케일 값을 곱하고, 정지 명령이 아닐 때만 적용합니다.
        pen_swing_stance_phase = swing_stance_penalty * ~self.no_commands * self.pen_scales["swingStancePhase"]
    
        # Foot Slip Penalty
        foot_xy_velocities = torch.norm(foot_velocities[:,:,:2],dim=-1)
        slip_err = torch.sum(self.foot_contact * foot_xy_velocities, dim=-1)
        pen_foot_slip = slip_err * self.pen_scales["footSlip"]

        # Foot Height Penalty
        ref_foot_height = 0.075
        foot_height_err = torch.sum(torch.abs(ref_foot_height - self.foot_pos[:, :, 2]), dim=-1)
        pen_foot_height = foot_height_err * torch.any(swing_phase, dim=-1) * ~self.no_commands * self.pen_scales["footHeight"]

        # Feet Air Time
        # '착지 순간' 감지: 이전 스텝엔 접촉 안 함(~) + 현재 스텝엔 접촉(&)
        # 실제 누적된 체공 시간과 목표 시간의 차이 계산
        # '착지 순간'에만 (foot_landing=True) 체공 시간 차이에 따른 보상/패널티 계산
        # (air_time_difference가 양수면 보상, 음수면 패널티)
        # 발이 공중에 떠 있을 때(~contact)만 체공 시간 누적
        # 발이 땅에 닿으면(contact) 체공 시간 리셋
        # 다음 스텝 계산을 위해 현재 접촉 상태 저장
        target_air_time = swing_ratio * self.cycle_time
        foot_landing = ~self.last_foot_contacts & self.foot_contact
        air_time_difference = self.foot_air_time - target_air_time
        rew_feet_air_time = torch.sum(air_time_difference * foot_landing, dim=-1) * self.rew_scales["feetAirTimeRewardScale"]
        self.foot_air_time[~self.foot_contact] += self.policy_dt
        self.foot_air_time[self.foot_contact] = 0.
        self.last_foot_contacts = self.foot_contact.clone()

        # === Leg Placement Penalties ===

        # 글로벌 좌표계에서 Base -> Foot 벡터 계산
        foot_pos_global_L = self.foot_pos[:, 0, :]
        foot_pos_global_R = self.foot_pos[:, 1, :]
        base_pos_global = self.root_states[:, :3]

        vec_base_to_foot_L_global = foot_pos_global_L - base_pos_global
        vec_base_to_foot_R_global = foot_pos_global_R - base_pos_global

        # 글로벌 벡터를 로컬 좌표계로 변환 (base_quat 사용)
        # base_quat는 post_physics_step에서 이미 계산됨
        vec_base_to_foot_L_local = quat_rotate_inverse(self.base_quat, vec_base_to_foot_L_global)
        vec_base_to_foot_R_local = quat_rotate_inverse(self.base_quat, vec_base_to_foot_R_global)

        # 로컬 Y 좌표를 사용하여 발 간격 계산
        foot_gap_y_local = torch.abs(vec_base_to_foot_L_local[:, 1] - vec_base_to_foot_R_local[:, 1])
        ref_foot_gap_y = 0.15
        foot_gap_y_err = torch.relu(foot_gap_y_local- ref_foot_gap_y)
        pen_footGapY = foot_gap_y_err * self.pen_scales["footGapY"]

        # Knee Y gap penalty
        # --- 로컬 좌표계에서 다리 간격 계산 ---
        leg_indices = self.knee_indices # TODO: Verify knee_indices point to Thigh/Calf
        leg_pos_global_L = self.rigid_body_pos[:, leg_indices[0], :]
        leg_pos_global_R = self.rigid_body_pos[:, leg_indices[1], :]

        vec_base_to_leg_L_global = leg_pos_global_L - base_pos_global
        vec_base_to_leg_R_global = leg_pos_global_R - base_pos_global

        vec_base_to_leg_L_local = quat_rotate_inverse(self.base_quat, vec_base_to_leg_L_global)
        vec_base_to_leg_R_local = quat_rotate_inverse(self.base_quat, vec_base_to_leg_R_global)

        leg_gap_y_local = torch.abs(vec_base_to_leg_L_local[:, 1] - vec_base_to_leg_R_local[:, 1])
        ref_leg_gap_y = 0.15 #0.20  
        leg_gap_y_err = torch.relu(leg_gap_y_local -ref_leg_gap_y)
        pen_legGapY = leg_gap_y_err * self.pen_scales["legGapY"]

        # Standing foot x gap penalty
        foot_gap_x_local_standing = torch.abs(vec_base_to_foot_L_local[:, 0] - vec_base_to_foot_R_local[:, 0])
        pen_footGapXStanding = foot_gap_x_local_standing * self.no_commands * self.pen_scales["footGapXStanding"]

        # === Unnecessary Joint Motion Penalties ===
        y_cmd_zero = self.commands[:, 1] == 0
        yaw_cmd_zero = self.commands[:, 2] == 0
        pen_hipYaw = torch.sum(torch.square(self.dof_pos[:, [0, 6]]), dim=-1) * y_cmd_zero * yaw_cmd_zero * self.pen_scales["hipYaw"]
        pen_hipRoll = torch.sum(torch.square(self.dof_pos[:, [1, 7]]), dim=-1) * y_cmd_zero * yaw_cmd_zero * self.pen_scales["hipRoll"]
        pen_ankleRoll = torch.sum(torch.square(self.dof_pos[:, [5, 11]]), dim=-1) * y_cmd_zero * yaw_cmd_zero * self.pen_scales["ankleRoll"]

        # === Safety & Limits Penalties ===
        # Knee Collision
        knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=-1) > 1.
        pen_Knee_collision = torch.sum(knee_contact, dim=1) * self.pen_scales["kneeCollisionRewardScale"]
        # Calf Collision
        calf_contact = torch.norm(self.contact_forces[:, self.calf_indices, :], dim=-1) > 1.
        pen_Calf_collision = torch.sum(calf_contact, dim=1) * self.pen_scales["calfCollisionRewardScale"]
        # Jump Penalty
        num_contact_feet = torch.sum(self.foot_contact, dim=-1)
        pen_jump = (num_contact_feet == 0) * self.pen_scales["jumpPenalty"]

        # (관절/액션/속도 한계 패널티 - 활성화 시 추가)
        softness = 0.9
        dof_pos_limit_lower = torch.sum(torch.relu(self.dof_pos_limits[:, 0].unsqueeze(0) * softness - self.dof_pos), dim=-1)
        dof_pos_limit_upper = torch.sum(torch.relu(self.dof_pos - self.dof_pos_limits[:, 1].unsqueeze(0) * softness), dim=-1)
        pen_dofLimit = (dof_pos_limit_lower + dof_pos_limit_upper) * self.pen_scales["dofLimit"]
        action_limit_lower = torch.sum(torch.relu(self.dof_pos_limits[:, 0].unsqueeze(0) * softness - (self.joint_actions + self.default_dof_pos)), dim=-1)
        action_limit_upper = torch.sum(torch.relu((self.joint_actions + self.default_dof_pos) - self.dof_pos_limits[:, 1].unsqueeze(0) * softness), dim=-1)
        pen_actionLimit = (action_limit_lower + action_limit_upper) * self.pen_scales["actionLimit"]
        dof_vel_limit_over = torch.relu(torch.abs(self.dof_vel) - self.dof_vel_limits.unsqueeze(0))
        pen_dofVelLimit = torch.sum(dof_vel_limit_over, dim=-1) * self.pen_scales["dofVelLimit"]

        
        # === Efficiency & Smoothness Penalties ===
        pen_torque = torch.sum(torch.square(self.torques), dim=1) * self.pen_scales["torqueRewardScale"]
        pen_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.pen_scales["jointAccRewardScale"]
        pen_action_rate = torch.where (torch.any(self.last_joint_actions != 0, dim=1),
                                      torch.sum(torch.square(self.last_joint_actions - self.joint_actions), dim=1) * self.pen_scales["actionRateRewardScale"],
                                      torch.zeros_like(self.last_joint_actions[:,0])
                                      )
        action_smoothness2_diff = torch.square(self.joint_actions - 2 * self.last_joint_actions + self.last_joint_actions2)
        pen_actionSmoothness2 = torch.where(torch.any(self.last_joint_actions != 0, dim=1) & torch.any(self.last_joint_actions2 != 0, dim=1) ,
                                           torch.sum(action_smoothness2_diff, dim=1) * self.pen_scales["actionSmoothness2RewardScale"],
                                           torch.zeros_like(self.last_joint_actions2[:,0])
                                           )
        joint_energy = torch.sum(torch.abs(self.torques * self.dof_vel), dim=-1)
        pen_jointEnergy = joint_energy * self.pen_scales["jointEnergy"]

        # === Cosmetics ===
        rew_hip = torch.sum(torch.abs(self.dof_pos[:, [1, 7]] - self.default_dof_pos[:, [1, 7]]), dim=1) * self.pen_scales["hipRewardScale"]
        default_pos_err = torch.square(self.default_dof_pos - self.dof_pos)
        pen_default_pos_standing = torch.sum(default_pos_err, dim=-1) * self.no_commands * self.pen_scales["defaultPosStanding"]


        # === 개별 보상 값 저장을 위한 코드 추가 ===
        self.reward_container.clear()
        reward_terms = [
            "rew_lin_vel_xy", "rew_ang_vel_z", "rew_feet_air_time", "rew_hip",
            "pen_lin_vel_z", "pen_ang_vel_xy", "pen_baseOrientation", "pen_baseHeight",
            "pen_swing_stance_phase", "pen_foot_slip", "pen_foot_height",
            "pen_footGapY", "pen_legGapY", "pen_footGapXStanding",
            "pen_hipYaw", "pen_hipRoll", "pen_ankleRoll",
            "pen_Knee_collision", "pen_Calf_collision", "pen_jump",
            "pen_dofLimit", "pen_actionLimit", "pen_dofVelLimit",
            "pen_torque", "pen_joint_acc", "pen_action_rate", "pen_actionSmoothness2", "pen_jointEnergy",
            "pen_default_pos_standing"
        ]

        for term in reward_terms:
             if term in locals():
                 self.reward_container[term] = locals()[term][self.observe_envs]

        # total reward
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_feet_air_time + rew_hip \
                     + pen_lin_vel_z + pen_ang_vel_xy + pen_baseOrientation + pen_baseHeight \
                     + pen_swing_stance_phase + pen_foot_slip + pen_foot_height \
                     + pen_footGapY + pen_legGapY + pen_footGapXStanding \
                     + pen_hipYaw + pen_hipRoll + pen_ankleRoll \
                     + pen_Knee_collision + pen_Calf_collision + pen_jump \
                     + pen_dofLimit + pen_actionLimit + pen_dofVelLimit \
                     + pen_torque + pen_joint_acc + pen_action_rate + pen_actionSmoothness2 + pen_jointEnergy \
                     + pen_default_pos_standing

        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # add termination reward
        self.rew_buf += self.rew_scales["terminalReward"] * self.reset_buf * ~self.timeout_buf #

        # log episode reward sums
        self.episode_sums["rew_lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["rew_ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["rew_feet_air_time"] += rew_feet_air_time
        self.episode_sums["rew_hip"] += rew_hip
        self.episode_sums["pen_lin_vel_z"] += pen_lin_vel_z
        self.episode_sums["pen_ang_vel_xy"] += pen_ang_vel_xy
        self.episode_sums["pen_baseOrientation"] += pen_baseOrientation
        self.episode_sums["pen_baseHeight"] += pen_baseHeight
        self.episode_sums["pen_swing_stance_phase"] += pen_swing_stance_phase
        self.episode_sums["pen_foot_slip"] += pen_foot_slip
        self.episode_sums["pen_foot_height"] += pen_foot_height
        self.episode_sums["pen_footGapY"] += pen_footGapY
        self.episode_sums["pen_legGapY"] += pen_legGapY
        self.episode_sums["pen_footGapXStanding"] += pen_footGapXStanding
        self.episode_sums["pen_hipYaw"] += pen_hipYaw
        self.episode_sums["pen_hipRoll"] += pen_hipRoll
        self.episode_sums["pen_ankleRoll"] += pen_ankleRoll
        self.episode_sums["pen_Knee_collision"] += pen_Knee_collision
        self.episode_sums["pen_Calf_collision"] += pen_Calf_collision
        self.episode_sums["pen_jump"] += pen_jump
        self.episode_sums["pen_dofLimit"] += pen_dofLimit
        self.episode_sums["pen_actionLimit"] += pen_actionLimit
        self.episode_sums["pen_dofVelLimit"] += pen_dofVelLimit
        self.episode_sums["pen_torque"] += pen_torque
        self.episode_sums["pen_joint_acc"] += pen_joint_acc
        self.episode_sums["pen_action_rate"] += pen_action_rate
        self.episode_sums["pen_actionSmoothness2"] += pen_actionSmoothness2
        self.episode_sums["pen_jointEnergy"] += pen_jointEnergy
        self.episode_sums["pen_default_pos_standing"] += pen_default_pos_standing


    def joy_callback(self, data):
        self.commands[:,0] = data.axes[1] * self.command_x_range[1]  # x vel
        self.commands[:,1] = data.axes[0] * self.command_y_range[1]  # y vel
        self.commands[:,2] = data.axes[3] * self.command_yaw_range[1]  # yaw vel

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

    def set_cmd(self, env_ids):
        # (env_ids에 해당하는 환경들에 대해 명령 생성)

        # 기본: 모든 명령을 랜덤 범위 내에서 생성
        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        # 작은 명령 값은 0으로 처리 (Deadzone)
        self.commands[env_ids, 0] = torch.where(torch.abs(self.commands[env_ids, 0]) <= 0.00, torch.tensor(0.0, device=self.device), self.commands[env_ids, 0])
        self.commands[env_ids, 1] = torch.where(torch.abs(self.commands[env_ids, 1]) <= 0.00, torch.tensor(0.0, device=self.device), self.commands[env_ids, 1])
        self.commands[env_ids, 2] = torch.where(torch.abs(self.commands[env_ids, 2]) <= 0.00, torch.tensor(0.0, device=self.device), self.commands[env_ids, 2])

        # 각 환경 그룹별로 명령 제한 적용
        # (환경 ID 필터링 및 해당 명령 요소 0 또는 특정 범위로 재설정)

        # 정지 환경 (Stand)
        stand_env_mask = (env_ids >= self.stand_env_range[0]) & (env_ids <= self.stand_env_range[1])
        stand_env_indices = env_ids[stand_env_mask]
        if len(stand_env_indices) > 0:
            self.commands[stand_env_indices, :] = 0.0 # 모든 명령 0

        # +X 환경
        plus_x_mask = (env_ids >= self.only_plus_x_envs_range[0]) & (env_ids <= self.only_plus_x_envs_range[1])
        plus_x_indices = env_ids[plus_x_mask]
        if len(plus_x_indices) > 0:
            self.commands[plus_x_indices, 0] = torch_rand_float(0, self.command_x_range[1], (len(plus_x_indices), 1), device=self.device).squeeze() # 0 이상
            self.commands[plus_x_indices, 1:] = 0.0 # Y, Yaw는 0

        # -X 환경
        minus_x_mask = (env_ids >= self.only_minus_x_envs_range[0]) & (env_ids <= self.only_minus_x_envs_range[1])
        minus_x_indices = env_ids[minus_x_mask]
        if len(minus_x_indices) > 0:
            self.commands[minus_x_indices, 0] = torch_rand_float(self.command_x_range[0], -0, (len(minus_x_indices), 1), device=self.device).squeeze() # -0 이하
            self.commands[minus_x_indices, 1:] = 0.0 # Y, Yaw는 0

        # +Y 환경
        plus_y_mask = (env_ids >= self.only_plus_y_envs_range[0]) & (env_ids <= self.only_plus_y_envs_range[1])
        plus_y_indices = env_ids[plus_y_mask]
        if len(plus_y_indices) > 0:
            self.commands[plus_y_indices, 1] = torch_rand_float(0, self.command_y_range[1], (len(plus_y_indices), 1), device=self.device).squeeze() # 0 이상
            self.commands[plus_y_indices, 0] = 0.0 # X는 0
            self.commands[plus_y_indices, 2] = 0.0 # Yaw는 0

        # -Y 환경
        minus_y_mask = (env_ids >= self.only_minus_y_envs_range[0]) & (env_ids <= self.only_minus_y_envs_range[1])
        minus_y_indices = env_ids[minus_y_mask]
        if len(minus_y_indices) > 0:
            self.commands[minus_y_indices, 1] = torch_rand_float(self.command_y_range[0], -0, (len(minus_y_indices), 1), device=self.device).squeeze() # -0 이하
            self.commands[minus_y_indices, 0] = 0.0 # X는 0
            self.commands[minus_y_indices, 2] = 0.0 # Yaw는 0

        # +Yaw 환경
        plus_yaw_mask = (env_ids >= self.only_plus_yaw_envs_range[0]) & (env_ids <= self.only_plus_yaw_envs_range[1])
        plus_yaw_indices = env_ids[plus_yaw_mask]
        if len(plus_yaw_indices) > 0:
            self.commands[plus_yaw_indices, 2] = torch_rand_float(0, self.command_yaw_range[1], (len(plus_yaw_indices), 1), device=self.device).squeeze() # 0 이상
            self.commands[plus_yaw_indices, 0] = 0.0 # X는 0
            self.commands[plus_yaw_indices, 1] = 0.0 # Y는 0

        # -Yaw 환경
        minus_yaw_mask = (env_ids >= self.only_minus_yaw_envs_range[0]) & (env_ids <= self.only_minus_yaw_envs_range[1])
        minus_yaw_indices = env_ids[minus_yaw_mask]
        if len(minus_yaw_indices) > 0:
            self.commands[minus_yaw_indices, 2] = torch_rand_float(self.command_yaw_range[0], -0, (len(minus_yaw_indices), 1), device=self.device).squeeze() # -0 이하
            self.commands[minus_yaw_indices, 0] = 0.0 # X는 0
            self.commands[minus_yaw_indices, 1] = 0.0 # Y는 0

        # +X +Yaw 환경
        px_pyaw_mask = (env_ids >= self.plus_x_plus_yaw_envs_range[0]) & (env_ids <= self.plus_x_plus_yaw_envs_range[1])
        px_pyaw_indices = env_ids[px_pyaw_mask]
        if len(px_pyaw_indices) > 0:
            self.commands[px_pyaw_indices, 0] = torch_rand_float(0, self.command_x_range[1], (len(px_pyaw_indices), 1), device=self.device).squeeze()
            self.commands[px_pyaw_indices, 1] = 0.0 # Y는 0
            self.commands[px_pyaw_indices, 2] = torch_rand_float(0, self.command_yaw_range[1], (len(px_pyaw_indices), 1), device=self.device).squeeze()

        # +X -Yaw 환경
        px_myaw_mask = (env_ids >= self.plus_x_minus_yaw_envs_range[0]) & (env_ids <= self.plus_x_minus_yaw_envs_range[1])
        px_myaw_indices = env_ids[px_myaw_mask]
        if len(px_myaw_indices) > 0:
            self.commands[px_myaw_indices, 0] = torch_rand_float(0, self.command_x_range[1], (len(px_myaw_indices), 1), device=self.device).squeeze()
            self.commands[px_myaw_indices, 1] = 0.0 # Y는 0
            self.commands[px_myaw_indices, 2] = torch_rand_float(self.command_yaw_range[0], 0, (len(px_myaw_indices), 1), device=self.device).squeeze()

        # -X +Yaw 환경
        mx_pyaw_mask = (env_ids >= self.minus_x_plus_yaw_envs_range[0]) & (env_ids <= self.minus_x_plus_yaw_envs_range[1])
        mx_pyaw_indices = env_ids[mx_pyaw_mask]
        if len(mx_pyaw_indices) > 0:
            self.commands[mx_pyaw_indices, 0] = torch_rand_float(self.command_x_range[0], 0, (len(mx_pyaw_indices), 1), device=self.device).squeeze()
            self.commands[mx_pyaw_indices, 1] = 0.0 # Y는 0
            self.commands[mx_pyaw_indices, 2] = torch_rand_float(0, self.command_yaw_range[1], (len(mx_pyaw_indices), 1), device=self.device).squeeze()

        # -X -Yaw 환경 (마지막 그룹)
        mx_myaw_mask = (env_ids >= self.minus_x_minus_yaw_envs_range[0]) & (env_ids <= self.minus_x_minus_yaw_envs_range[1])
        mx_myaw_indices = env_ids[mx_myaw_mask]
        if len(mx_myaw_indices) > 0:
            self.commands[mx_myaw_indices, 0] = torch_rand_float(self.command_x_range[0], 0, (len(mx_myaw_indices), 1), device=self.device).squeeze()
            self.commands[mx_myaw_indices, 1] = 0.0 # Y는 0
            self.commands[mx_myaw_indices, 2] = torch_rand_float(self.command_yaw_range[0], 0, (len(mx_myaw_indices), 1), device=self.device).squeeze() 


        # 위에서 처리되지 않은 모든 env_ids에 해당
        # 이미 랜덤 명령이 할당되어 있으므로, 특별히 수정할 필요 없음
        # (만약 이 그룹에도 특정 명령만 주고 싶다면 위와 유사하게 마스크 생성 및 재할당)

    def freeze(self):
        """Sets commands to zero for all environments."""
        self.commands[:, 0] = 0.0 # X velocity
        self.commands[:, 1] = 0.0 # Y velocity
        self.commands[:, 2] = 0.0 # Yaw velocity

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
            self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # # self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1) # set small commands to zero

        self.set_cmd(env_ids)

        self.last_joint_actions[env_ids] = 0.
        self.last_joint_actions2[env_ids] = 0.
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
            self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
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
        self.root_states[:, 7:9] = torch_rand_float(-0.5, 0.5, (self.num_envs, 2), device=self.device) # lin vel x/y
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

        # print("joint_actions_raw:", joint_actions_raw[0])
        # print("self.joint_actions:", self.joint_actions[0])
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
            # torques = torch.clip(torques, -300., 300.)
            torques = torch.clip(torques, -self.joint_torque_limits, self.joint_torque_limits)

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
        if (self.common_step_counter % self.push_interval == 0) and not self.test_mode:
            self.push_robots()

        # 주기적으로 freeze 상태 시작
        if self.common_step_counter % self.freeze_interval == 0 and not self.test_mode:
            self.freeze()        # 모든 환경 명령 0으로 설정
            self.freeze_flag = True  # freeze 상태 플래그 켜기
            # freeze 지속 시간 랜덤화 (선택적)
            self.freeze_steps = np.random.randint(150, 300)
            # self.freeze_steps = 150 # 우선 고정값 사용 (약 1.2초 = 150 * 0.008)

        # freeze 상태 지속 및 해제
        if self.freeze_flag:
            self.freeze_cnt += 1 # freeze 카운터 증가
            if self.freeze_cnt >= self.freeze_steps: # 설정된 시간이 지나면
                all_env_ids = torch.arange(self.num_envs, device=self.device)
                self.set_cmd(all_env_ids) # 모든 환경에 다시 일반 명령 할당
                self.freeze_flag = False    # freeze 상태 플래그 끄기
                self.freeze_cnt = 0         # freeze 카운터 리셋

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
        
        self.last_joint_actions2[:] = self.last_joint_actions[:]
        self.last_joint_actions[:] = self.joint_actions[:]

        self.last_dof_vel[:] = self.dof_vel[:]

        self.last_clock_actions[:] = self.clock_actions[:] 

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

        base_pos_msg = Point()
        base_pos_msg.x = self.root_states[env_id, 0].item() # Base X position
        base_pos_msg.y = self.root_states[env_id, 1].item() # Base Y position
        base_pos_msg.z = self.root_states[env_id, 2].item() # Base Z position
        topic_name_pos = '/rok4/base/position'
        if topic_name_pos not in self.ros_publishers:
            self.ros_publishers[topic_name_pos] = rospy.Publisher(topic_name_pos, Point, queue_size=10)
        self.ros_publishers[topic_name_pos].publish(base_pos_msg)
        # --- ↑↑↑ ---

        # --- ↓↓↓ 베이스 방향(쿼터니언) 발행 코드 추가 ---
        base_ori_msg = Quaternion()
        base_ori_msg.x = self.root_states[env_id, 3].item() # Quaternion X
        base_ori_msg.y = self.root_states[env_id, 4].item() # Quaternion Y
        base_ori_msg.z = self.root_states[env_id, 5].item() # Quaternion Z
        base_ori_msg.w = self.root_states[env_id, 6].item() # Quaternion W
        topic_name_ori = '/rok4/base/orientation_quat'
        if topic_name_ori not in self.ros_publishers:
            self.ros_publishers[topic_name_ori] = rospy.Publisher(topic_name_ori, Quaternion, queue_size=10)
        self.ros_publishers[topic_name_ori].publish(base_ori_msg)

        # 쿼터니언 가져오기 (단일 환경 데이터)
        quat_np = self.root_states[env_id, 3:7].cpu().numpy() # Scipy는 NumPy 배열 입력
        
        # Scipy Rotation 객체 생성 (x, y, z, w 순서 확인!)
        # Isaac Gym 쿼터니언은 [x, y, z, w] 순서
        rotation = R.from_quat(quat_np)

        # ZYX 오일러 각도 (intrinsic) 계산 (라디안 단위)
        # 'zyx' 순서로 지정, degrees=False (기본값)
        euler_zyx = rotation.as_euler('zyx')

        yaw = euler_zyx[0]   # Z 회전 (Yaw)
        pitch = euler_zyx[1] # Y 회전 (Pitch)
        roll = euler_zyx[2]  # X 회전 (Roll)

        # Vector3 메시지에 ZYX 순서로 저장 (msg.z = Yaw, msg.y = Pitch, msg.x = Roll)
        base_euler_msg = Vector3()
        base_euler_msg.z = yaw    # Yaw (Z 회전)
        base_euler_msg.y = pitch  # Pitch (Y 회전)
        base_euler_msg.x = roll   # Roll (X 회전)

        topic_name_euler = '/rok4/base/orientation_euler_zyx'
        if topic_name_euler not in self.ros_publishers:
            self.ros_publishers[topic_name_euler] = rospy.Publisher(topic_name_euler, Vector3, queue_size=10)
        self.ros_publishers[topic_name_euler].publish(base_euler_msg)


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
            
        # Clock Actions (phi_L, phi_R, delta_L, delta_R)
        clock_actions_msg = Float32MultiArray()
        clock_actions_msg.data = [
            self.clock_actions[env_id, 0].item(), # phi_L
            self.clock_actions[env_id, 1].item(), # phi_R
            self.clock_actions[env_id, 2].item(), # delta_L
            self.clock_actions[env_id, 3].item()  # delta_R
        ]
        topic_name_ca = '/rok4/gait/clock_actions_array' # 토픽 이름 변경
        if topic_name_ca not in self.ros_publishers:
            self.ros_publishers[topic_name_ca] = rospy.Publisher(topic_name_ca, Float32MultiArray, queue_size=10)
        self.ros_publishers[topic_name_ca].publish(clock_actions_msg)

        # Cycle Time (t_L, t_R)
        cycle_t_msg = Float32MultiArray()
        cycle_t_msg.data = [
            self.cycle_t[env_id, 0].item(), # t_L
            self.cycle_t[env_id, 1].item()  # t_R
        ]
        topic_name_ct = '/rok4/gait/cycle_t_array' # 토픽 이름 변경
        if topic_name_ct not in self.ros_publishers:
            self.ros_publishers[topic_name_ct] = rospy.Publisher(topic_name_ct, Float32MultiArray, queue_size=10)
        self.ros_publishers[topic_name_ct].publish(cycle_t_msg)

        # Sin Cycle (sin_L, sin_R)
        sin_cycle_msg = Float32MultiArray()
        sin_cycle_msg.data = [
            self.sin_cycle[env_id, 0].item(), # sin_L
            self.sin_cycle[env_id, 1].item()  # sin_R
        ]
        topic_name_sin = '/rok4/gait/sin_cycle_array' # 토픽 이름 변경
        if topic_name_sin not in self.ros_publishers:
            self.ros_publishers[topic_name_sin] = rospy.Publisher(topic_name_sin, Float32MultiArray, queue_size=10)
        self.ros_publishers[topic_name_sin].publish(sin_cycle_msg)

        # Cos Cycle (cos_L, cos_R)
        cos_cycle_msg = Float32MultiArray()
        cos_cycle_msg.data = [
            self.cos_cycle[env_id, 0].item(), # cos_L
            self.cos_cycle[env_id, 1].item()  # cos_R
        ]
        topic_name_cos = '/rok4/gait/cos_cycle_array' # 토픽 이름 변경
        if topic_name_cos not in self.ros_publishers:
            self.ros_publishers[topic_name_cos] = rospy.Publisher(topic_name_cos, Float32MultiArray, queue_size=10)
        self.ros_publishers[topic_name_cos].publish(cos_cycle_msg)

        # --- 개별 보상 항목 ---
        for reward_name, reward_value in self.reward_container.items():
            if reward_name.startswith("rew_"):
                topic_name = f'/rok4/reward/{reward_name}'
            elif reward_name.startswith("pen_"):
                 topic_name = f'/rok4/penalty/{reward_name}'
            else:
                 topic_name = f'/rok4/other/{reward_name}' # 혹시 모를 다른 이름 처리


            if topic_name not in self.ros_publishers:
                self.ros_publishers[topic_name] = rospy.Publisher(topic_name, Float32, queue_size=10)
            reward_msg = Float32()

            # reward_value는 이미 self.policy_dt가 곱해진 값이므로, 원래 스케일을 보려면 dt로 나눠줍니다.
            if self.policy_dt > 0: # dt가 0이 아닐 때만 나눗셈 수행
                reward_msg.data = reward_value.item() / self.policy_dt
            else:
                reward_msg.data = reward_value.item()


            # print("self.policy_dt:", self.policy_dt)
            # reward_msg.data = reward_value.item() # / self.policy_dt # 필요시 dt로 나누어 원래 스케일 확인
            self.ros_publishers[topic_name].publish(reward_msg)

        # --- 총 보상 ---
        total_reward_msg = Float32()
        if self.policy_dt > 0:
            total_reward_msg.data = self.rew_buf[env_id].item() / self.policy_dt # env_id 사용
        else:
            total_reward_msg.data = self.rew_buf[env_id].item() # env_id 사용
        if 'total_reward' in self.ros_publishers:
            self.ros_publishers['total_reward'].publish(total_reward_msg)
    # =================================================

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
