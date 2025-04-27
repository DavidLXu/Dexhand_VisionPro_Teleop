# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os.path as osp
import os, glob, tqdm
import random, torch, trimesh

from isaacgym import gymtorch
from isaacgym import gymapi

from utils.general_utils import *
from utils.torch_jit_utils import *
# from utils.hand_model import ShadowHandModel
from utils.render_utils import PytorchBatchRenderer

from sklearn.decomposition import PCA
from tasks.hand_base.base_task import BaseTask

import transforms3d

from utils.avp_leap import AllegroPybulletIKPython

sys.path.append(osp.join(BASE_DIR, 'dexgrasp/autoencoding'))
from dexgrasp.autoencoding.PN_Model import AutoencoderPN, AutoencoderTransPN

from typing import List, Optional, Union, Tuple

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pyspacemouse
import pygame
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)

class Compose:
    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript.

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self, rotation: Union[torch.Tensor, np.ndarray], convention: str = "xyz", **kwargs):
        convention = convention.lower()
        if not (set(convention) == set("xyz") and len(convention) == 3):
            raise ValueError(f"Invalid convention {convention}.")
        if isinstance(rotation, np.ndarray):
            data_type = "numpy"
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = "tensor"
        else:
            raise TypeError("Type of rotation should be torch.Tensor or numpy.ndarray")
        for t in self.transforms:
            if "convention" in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == "numpy":
            rotation = rotation.detach().cpu().numpy()
        return rotation

def rot6d_to_aa(rotation_6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to axis angles.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Invalid input rotation_6d f{rotation_6d.shape}.")
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle])
    return t(rotation_6d)

def quat_to_rotmat(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation matrixs.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions shape f{quaternions.shape}.")
    t = Compose([quaternion_to_matrix])
    return t(quaternions)


def quat_to_rot6d(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert quaternions to rotation 6d representations.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid input quaternions f{quaternions.shape}.")
    t = Compose([quaternion_to_matrix, matrix_to_rotation_6d])
    return t(quaternions)

def euler_to_quat(euler_angles: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    t = Compose([euler_angles_to_matrix, matrix_to_quaternion])
    return t(euler_angles)


class KeyboardTeleopDevice():
    def __init__(self, num_envs, device, gym, viewer):
        self.num_envs = num_envs
        self.device = device
        self.gym = gym
        self.viewer = viewer

        '''
        In the gym viewer init viewer angle, +x is pointint left, +y is pointing at us, +z is pointing up.
        Below shows the keymap for teleoperation.
        q(-z) w(-y) e(+z)                       u(+qy) i(+qx) o(-qy)
        a(+x) s(+y) d(-x) f(grasp)   h(release) j(+qz) k(-qx) l(-qz)
        '''

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "y_minus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "y_plus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "x_plus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "x_minus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "z_minus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "z_plus")
        # note the rotation action name here is not correct, refer to the comment above
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "rotz_minus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "rotz_plus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "roty_plus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "roty_minus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U, "rotx_plus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "rotx_minus")


        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "grasp")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_H, "release")
        
        # Initialize position and euler angles
        self.position = torch.tensor([0, 0, 0.5], device=device)
        self.euler_angles = torch.tensor([1.57, 0, 1.57], device=device)
        self.position_step = 0.05
        self.rotation_step = 0.157*2

        self.finger_val = torch.zeros((self.num_envs, 16), device=device)
        
    def get_teleop_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        for evt in self.gym.query_viewer_action_events(self.viewer):
            # print("action name", evt.action)
            # controlling translation
            if evt.action == "x_minus" and evt.value > 0:
                self.position[0] -= self.position_step
                
            if evt.action == "y_minus" and evt.value > 0:
                self.position[1] -= self.position_step

            if evt.action == "z_minus" and evt.value > 0:
                self.position[2] -= self.position_step

            if evt.action == "x_plus" and evt.value > 0:
                self.position[0] += self.position_step

            if evt.action == "y_plus" and evt.value > 0:
                self.position[1] += self.position_step

            if evt.action == "z_plus" and evt.value > 0:
                self.position[2] += self.position_step

            # controlling rotation
            if evt.action == "rotx_minus" and evt.value > 0:
                self.euler_angles[0] -= self.rotation_step

            if evt.action == "roty_minus" and evt.value > 0:
                self.euler_angles[1] -= self.rotation_step

            if evt.action == "rotz_minus" and evt.value > 0:
                self.euler_angles[2] -= self.rotation_step

            if evt.action == "rotx_plus" and evt.value > 0:
                self.euler_angles[0] += self.rotation_step

            if evt.action == "roty_plus" and evt.value > 0:
                self.euler_angles[1] += self.rotation_step

            if evt.action == "rotz_plus" and evt.value > 0:
                self.euler_angles[2] += self.rotation_step

            # # grasping
            if evt.action == "grasp" and evt.value > 0:
                self.finger_val[:,:] = 1.2
                self.finger_val[:,0] = 0
                self.finger_val[:,4] = 0
                self.finger_val[:,8] = 0 
            # # releasing
            if evt.action == "release" and evt.value > 0:
                self.finger_val[:,:] = 0
        

        # Clamp euler angles to prevent unstable rotations
        # self.euler_angles = torch.clamp(self.euler_angles, -3.14159, 3.14159)

        # Convert euler angles to quaternion
        target_quat = euler_to_quat(self.euler_angles.unsqueeze(0))

        # Prepare outputs
        target_pos = self.position
        right_hand_target_pos = target_pos.repeat(self.num_envs, 1)
        right_hand_target_rot = target_quat.repeat(self.num_envs, 1)
        right_hand_target_finger = self.finger_val

        return right_hand_target_pos, right_hand_target_rot, right_hand_target_finger

class SpaceMouseTeleop:
    def __init__(self, num_envs: int, device: str):
        """Initialize SpaceMouse teleop interface"""
        self.num_envs = num_envs
        self.device = device
        
        # Initialize position and euler angles
        self.position = torch.tensor([0, 0, 0.5], device=device)
        self.euler_angles = torch.tensor([1.57, 0, 1.57], device=device)
        
        # Movement sensitivity
        self.position_step = 0.05  
        self.rotation_step = 0.157*2
        
        # Initialize finger values
        self.finger_val = torch.zeros((self.num_envs, 16), device=device)
        
        # Connect to SpaceMouse
        success = pyspacemouse.open()
        if not success:
            print("Failed to connect to SpaceMouse")
            
    def get_teleop_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get teleop data from SpaceMouse"""
        
        # Read SpaceMouse state
        state = pyspacemouse.read()
        
        if state:
            # Update position based on translation
            self.position[0] += state.x * self.position_step
            self.position[1] += state.y * self.position_step 
            self.position[2] += state.z * self.position_step
            
            # Update rotation based on roll/pitch/yaw
            self.euler_angles[0] += state.roll * self.rotation_step
            self.euler_angles[2] += state.pitch * self.rotation_step
            self.euler_angles[1] -= state.yaw * self.rotation_step
            
            # Handle button presses for grasping
            if state.buttons[0]:  # Left button press
                self.finger_val[:,:] = 1.2
                self.finger_val[:,0] = 0
                self.finger_val[:,4] = 0
                self.finger_val[:,8] = 0
            elif state.buttons[1]:  # Right button press
                self.finger_val[:,:] = 0
            # Wrap euler angles to -pi to pi
            self.euler_angles = torch.remainder(self.euler_angles + torch.pi, 2*torch.pi) - torch.pi
        # Convert euler angles to quaternion
        target_quat = euler_to_quat(self.euler_angles.unsqueeze(0))

        # Prepare outputs
        target_pos = self.position
        right_hand_target_pos = target_pos.repeat(self.num_envs, 1)
        right_hand_target_rot = target_quat.repeat(self.num_envs, 1)
        right_hand_target_finger = self.finger_val

        return right_hand_target_pos, right_hand_target_rot, right_hand_target_finger

   
class JoystickTeleopDevice():
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device

        # Initialize position and euler angles
        self.position = torch.tensor([0, 0, 0.5], device=device)
        self.euler_angles = torch.tensor([1.57, 0, 1.57], device=device)
        self.position_step = 0.01
        self.rotation_step = 0.157

        self.finger_val = torch.zeros((self.num_envs, 16), device=device)

        # Try to initialize joystick
        try:
            
            pygame.init()
            pygame.joystick.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print("Joystick connected successfully!")
            self.has_joystick = True
        except:
            print("No joystick detected")
            self.has_joystick = False

    def get_teleop_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get teleop data from joystick"""
        if self.has_joystick:
            # Process joystick events
            pygame.event.pump()
            # Get joystick axes values (-1 to 1)
            # Apply deadzone threshold
            deadzone = 0.05
            
            # Left stick controls XY translation
            x_axis = self.joystick.get_axis(0)
            y_axis = self.joystick.get_axis(1)
            if abs(x_axis) > deadzone:
                self.position[0] += -x_axis * self.position_step  # X translation
            if abs(y_axis) > deadzone:
                self.position[1] += y_axis * self.position_step # Y translation
            
            # Right stick controls rotation, X/Y buttons control Z translation
            if self.joystick.get_button(3):  # X button
                self.position[2] -= self.position_step  # Z down
            if self.joystick.get_button(4):  # Y button
                self.position[2] += self.position_step  # Z up

            # Apply deadzone to rotation controls
            roll = self.joystick.get_axis(2)
            pitch = self.joystick.get_axis(3)
            if abs(roll) > deadzone:
                self.euler_angles[0] += -roll * self.rotation_step # Roll
            if abs(pitch) > deadzone:
                self.euler_angles[2] += pitch * self.rotation_step # Pitch

            # Shoulder triggers control yaw
            left_trigger = self.joystick.get_axis(4)
            right_trigger = self.joystick.get_axis(5)
            if abs(left_trigger) > deadzone or abs(right_trigger) > deadzone:
                yaw_control = -(left_trigger - right_trigger) * 0.5
                self.euler_angles[1] += yaw_control * self.rotation_step

            # Handle buttons for grasping
            if self.joystick.get_button(0):  # A button
                self.finger_val[:,:] = 1.2
                self.finger_val[:,0] = 0
                self.finger_val[:,4] = 0  
                self.finger_val[:,8] = 0
            elif self.joystick.get_button(1):  # B button
                self.finger_val[:,:] = 0

            # Wrap euler angles to -pi to pi
            self.euler_angles = torch.remainder(self.euler_angles + torch.pi, 2*torch.pi) - torch.pi

        # Convert euler angles to quaternion
        target_quat = euler_to_quat(self.euler_angles.unsqueeze(0))

        # Prepare outputs
        target_pos = self.position
        right_hand_target_pos = target_pos.repeat(self.num_envs, 1)
        right_hand_target_rot = target_quat.repeat(self.num_envs, 1)
        right_hand_target_finger = self.finger_val

        return right_hand_target_pos, right_hand_target_rot, right_hand_target_finger


# DexhandTeleop Single Hand Teleop Env
class DexhandTeleop(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=None, is_multi_agent=False):
        # init task setting
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.init_hand_dof = [0.0 for _ in range(16)]

        if self.cfg['env']['use_xarm6']:
            init_arm_dof = [0,0,0,-3.14,1.57,3.14] 
            self.init_hand_dof =  init_arm_dof  + self.init_hand_dof

        self.init_dof_state =  self.init_hand_dof

        self.HAND_BIT = 0b001     # 0001 in binary

        self.pos_error_integral = 0
        self.prev_pos_error = 0
        self.rot_error_integral = 0
        self.prev_rot_error = 0

        # load train/test config: modes and weights
        self.algo = cfg['algo']
        self.config = cfg['config']

        # init params from cfg
        self.init_wandb = self.cfg["wandb"]
        self.start_line, self.end_line, self.group = self.cfg["start_line"], self.cfg["end_line"], self.cfg["group"]
        self.shuffle_dict, self.shuffle_env = self.cfg['shuffle_dict'], self.cfg['shuffle_env']

        # # Run params
        self.up_axis = 'z'
        self.num_envs = self.cfg["env"]["numEnvs"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.is_testing, self.test_epoch, self.test_iteration, self.current_test_iteration = cfg['test'], cfg['test_epoch'], self.cfg["test_iteration"], 0
        self.current_iteration = 0
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.envs = []

        # # Control frequency
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        print("Control frequency inverse: ", control_freq_inv)

        # index, middle, ring, thumb
        self.fingertips = ['link_3.0_tip', 'link_11.0_tip', "link_15.0_tip", "link_7.0_tip"]

        self.num_fingertips = len(self.fingertips) 

        self.num_agents = 1
        self.cfg["env"]["numActions"] = 22
        # # Device params
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["graphics_device_id"] = device_id
        # # Visualize params
        self.cfg["headless"] = headless

        
        # # Render settings
        self.render_each_view = self.cfg["render_each_view"]
        self.render_hyper_view = self.cfg["render_hyper_view"]
        # init render folder and render_env_list
        self.render_folder = None
        self.render_env_list = list(range(9)) if self.render_hyper_view else None

        
        # default init from BaseTask create gym, sim, viewer, buffers for obs and states
        super().__init__(cfg=self.cfg, enable_camera_sensors=True if (headless or self.render_each_view or self.render_hyper_view) else False)

        # set viewer camera pose
        self.look_at_env = None
        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.0, 1.0, 0.9)
            cam_target = gymapi.Vec3(0, 0, 0.1)
            self.look_at_env = self.envs[len(self.envs) // 2]
            self.gym.viewer_camera_look_at(self.viewer, self.look_at_env, cam_pos, cam_target)


        # acquire gym tensors
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        net_contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "hand")
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        # wrap gym tensors to PyTorch tensors
        self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) 
        self.net_contact_force = gymtorch.wrap_tensor(net_contact_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dexterous_hand_dofs)

        # refresh tensor
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.dof_force_tensor = self.dof_force_tensor[:, :self.num_dexterous_hand_dofs]

        # dexterous_hand_dof
        self.z_theta = torch.zeros(self.num_envs, device=self.device)
        # self.dexterous_hand_default_dof_pos = torch.zeros(self.num_dexterous_hand_dofs, dtype=torch.float, device=self.device)
        self.dexterous_hand_default_dof_pos = torch.tensor(self.init_dof_state, dtype=torch.float, device=self.device)

        
        self.dexterous_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dexterous_hand_dofs]
        self.dexterous_hand_dof_pos = self.dexterous_hand_dof_state[..., 0]
        self.dexterous_hand_dof_vel = self.dexterous_hand_dof_state[..., 1]
   

        self.num_bodies = self.rigid_body_states.shape[1]

        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        # control tensor
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # NOTE the default init pose for arm and hand
        self.cur_targets = torch.tensor(self.init_dof_state, dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        # utility tensor

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.final_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # debug tensor
        self.right_hand_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.right_hand_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        #! modify ip address
        # self.teleop_ik = AllegroPybulletIKPython("192.168.100.17") 
        # self.teleop_device = KeyboardTeleopDevice(self.num_envs, self.device, self.gym, self.viewer)
        self.teleop_device = JoystickTeleopDevice(self.num_envs, self.device)


    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        # create sim following BaseTask
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # create ground plane
        self._create_ground_plane()
        # create envs
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):

        # # # ---------------------- Init Settings ---------------------- # #
        # locate Asset path
        self.assets_path = self.cfg["env"]['asset']['assetRoot']
        if not osp.exists(self.assets_path): self.assets_path = '../' + self.assets_path

        # load dexterous_hand asset
        dexterous_hand_asset, dexterous_hand_start_pose, dexterous_hand_dof_props = self._load_dexterous_hand_assets(self.assets_path)

        self.env_object_scale = []
        self.hand_indices= []


        # # ---------------------- Create Envs ---------------------- # #
        print('Create num_envs', self.num_envs)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        # create envs
        loop = tqdm.tqdm(range(self.num_envs))
        for env_id in loop:
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            loop.set_description('Creating env {}/{}'.format(env_id, self.num_envs))
            
            # # ---------------------- Create ShadowHand Actor ---------------------- # #

            dexterous_hand_actor = self.gym.create_actor(env_ptr, 
                dexterous_hand_asset, 
                dexterous_hand_start_pose, 
                "hand", 
                env_id, 
                self.HAND_BIT, 
                3
            )

            dexterous_hand_dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
            if self.cfg['env']['use_xarm6']:
                dexterous_hand_dof_props["stiffness"] =[1500] * 22
                dexterous_hand_dof_props["damping"] = [10] * 22
            else:
                dexterous_hand_dof_props["stiffness"] =[1500] * 16
                dexterous_hand_dof_props["damping"] = [10] * 16

            self.gym.set_actor_dof_properties(env_ptr, dexterous_hand_actor, dexterous_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, dexterous_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # stop aggregate actors
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            # append env_ptr
            self.envs.append(env_ptr)
        
        # send items to torch
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        
    def _load_dexterous_hand_assets(self, assets_path):        

        if self.cfg['env']['use_xarm6']:
            dexterous_hand_asset_file = "xarm6/xarm6_allegro_right.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.flip_visual_attachments = False
            asset_options.fix_base_link = True
            asset_options.collapse_fixed_joints = False
            asset_options.disable_gravity = True
            asset_options.thickness = 0.001
            asset_options.angular_damping = 20 # consisent with ManipTrans
            asset_options.linear_damping = 20 # consisent with ManipTrans
        else:

            dexterous_hand_asset_file = "allegro_hand/allegro_hand_right_glb.urdf"
            # dexterous_hand_asset_file = "allegro_hand/allegro_hand_right_glb_original_inertia.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.flip_visual_attachments = False
            asset_options.fix_base_link = False
            asset_options.collapse_fixed_joints = False
            asset_options.disable_gravity = True
            asset_options.thickness = 0.001
            asset_options.angular_damping = 20 # consisent with ManipTrans
            asset_options.linear_damping = 20 # consisent with ManipTrans

        # set dexterous_hand AssetOptions
        
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        
        # load dexterous_hand_asset
        dexterous_hand_asset = self.gym.load_asset(self.sim, assets_path, dexterous_hand_asset_file, asset_options)
        self.num_dexterous_hand_bodies = self.gym.get_asset_rigid_body_count(dexterous_hand_asset)  # 24
        self.num_dexterous_hand_shapes = self.gym.get_asset_rigid_shape_count(dexterous_hand_asset)  # 20
        self.num_dexterous_hand_dofs = self.gym.get_asset_dof_count(dexterous_hand_asset)  # 22

        if self.cfg.get('env', {}).get('print_asset_info', False):
            # for debug use
            for i in range(self.num_dexterous_hand_bodies):
                print(f"Body {i}: {self.gym.get_asset_rigid_body_name(dexterous_hand_asset, i)}")

            for i in range(self.num_dexterous_hand_dofs):
                print(f"Joint {i}: {self.gym.get_asset_dof_name(dexterous_hand_asset, i)}")

        # valid rigid body indices # NOTE 8 is the starting index of hand
        self.valid_dexterous_hand_bodies = [i for i in range(8,self.num_dexterous_hand_bodies)]


        # set dexterous_hand dof properties
        dexterous_hand_dof_props = self.gym.get_asset_dof_properties(dexterous_hand_asset)
        self.dexterous_hand_dof_lower_limits = []
        self.dexterous_hand_dof_upper_limits = []
        self.dexterous_hand_dof_default_pos = []
        self.dexterous_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_dexterous_hand_dofs):
            self.dexterous_hand_dof_lower_limits.append(dexterous_hand_dof_props['lower'][i])
            self.dexterous_hand_dof_upper_limits.append(dexterous_hand_dof_props['upper'][i])
            self.dexterous_hand_dof_default_pos.append(0.0)
            self.dexterous_hand_dof_default_vel.append(0.0)

        self.dexterous_hand_dof_lower_limits = to_torch(self.dexterous_hand_dof_lower_limits, device=self.device)
        self.dexterous_hand_dof_upper_limits = to_torch(self.dexterous_hand_dof_upper_limits, device=self.device)
        self.dexterous_hand_dof_default_pos = to_torch(self.dexterous_hand_dof_default_pos, device=self.device)
        self.dexterous_hand_dof_default_vel = to_torch(self.dexterous_hand_dof_default_vel, device=self.device)

        if self.cfg['env']['use_xarm6']:
            hand_shift_x = 0 #-0.35
            hand_shift_y = 0.65
            hand_shift_z = 0.0
        else:
            hand_shift_x = 0
            hand_shift_y = 0.0
            hand_shift_z = 0.4

        dexterous_hand_start_pose = gymapi.Transform()
        dexterous_hand_start_pose.p = gymapi.Vec3(hand_shift_x, hand_shift_y, hand_shift_z)  
        if self.cfg['env']['use_xarm6']:
            dexterous_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, -1.57)  
        else:
            dexterous_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)   


        # locate hand body index   
        body_names = {'wrist': 'base_link', 'palm': 'base_link', 'thumb': 'link_7.0_tip',
                      'index': 'link_3.0_tip', 'middle': 'link_11.0_tip', 'ring': 'link_15.0_tip'}
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(dexterous_hand_asset, body_name)

        # locate fingertip_handles indices [5, 9, 13, 18, 23]
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(dexterous_hand_asset, name) for name in self.fingertips]

        return dexterous_hand_asset, dexterous_hand_start_pose, dexterous_hand_dof_props

    # # ---------------------- Physics Simulation Steps ---------------------- # #
    def keyboard_action(self,actions):
        right_hand_target_pos, right_hand_target_rot, right_hand_target_finger = self.teleop_device.get_teleop_data()

        position_error = (right_hand_target_pos - self.right_hand_pos)
        self.pos_error_integral += position_error * self.dt
        self.pos_error_integral = torch.clamp(self.pos_error_integral, -1, 1)
        pos_derivative = (position_error - self.prev_pos_error) / self.dt
        self.Kp_pos = 4000
        self.Ki_pos = 0.05
        self.Kd_pos = 5
        force = self.Kp_pos * position_error + self.Ki_pos * self.pos_error_integral + self.Kd_pos * pos_derivative
        self.apply_forces[:, 0, :] = force 


        rotation_error = orientation_error(right_hand_target_rot, self.right_hand_rot)
        self.rot_error_integral += rotation_error * self.dt
        self.rot_error_integral = torch.clamp(self.rot_error_integral, -1, 1)
        rot_derivative = (rotation_error - self.prev_rot_error) / self.dt
        self.Kp_rot = 40 # 0.3
        self.Ki_rot = 0.001
        self.Kd_rot = 1.5 # 0.005
        torque = self.Kp_rot * rotation_error + self.Ki_rot * self.rot_error_integral + self.Kd_rot * rot_derivative
        self.apply_torque[:, 1, :] = torque

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        self.cur_targets[:, :] = right_hand_target_finger
        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.cur_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
        self.prev_pos_error = position_error
        self.prev_rot_error = rotation_error
        self.prev_targets = self.cur_targets  



    def armless_hand_action(self, actions):
        #! vision_pro_teleop
        avp_output_joints, avp_output_wrist = self.teleop_ik.get_avp_data()
        avp_output_wrist = avp_output_wrist.to(self.device)
        

        #* armless position control
        
        avp_output_wrist[:,0] = -avp_output_wrist[:,0]  # x -> -x
        avp_output_wrist[:,1] = -avp_output_wrist[:,1]  # y -> -y
        
       

        hand_pose_ref = [0,0,0,0,0,0]
        hand_pose_ref[0:3] = avp_output_wrist.squeeze(0)[:3].to(self.device)
        hand_pose_ref[1] += 0.4
        hand_pose_ref[2] -= 0.85 # seat height for teleop
        
        right_hand_target_pos = torch.tensor(
            [hand_pose_ref[0], hand_pose_ref[1], hand_pose_ref[2]+ 0.1153], 
            device=self.device,dtype=torch.float
        )
        right_hand_target_pos = right_hand_target_pos.repeat(self.num_envs, 1)

        position_error = (right_hand_target_pos - self.right_hand_pos)
        self.pos_error_integral += position_error * self.dt
        self.pos_error_integral = torch.clamp(self.pos_error_integral, -1, 1)
        pos_derivative = (position_error - self.prev_pos_error) / self.dt
        self.Kp_pos = 4000
        self.Ki_pos = 0.05
        self.Kd_pos = 5
        #! for urdf with original inertia
        # self.Kp_pos = 200
        # self.Ki_pos = 0.05
        # self.Kd_pos = 0.02
        force = self.Kp_pos * position_error + self.Ki_pos * self.pos_error_integral + self.Kd_pos * pos_derivative

        self.apply_forces[:, 0, :] = force 
        
        #* armless orientation control
        wrist_rot_quat = avp_output_wrist[:,3:7].to(self.device)
        z_rot_180 = torch.tensor([0, 0, 1, 0], device=self.device).repeat(avp_output_wrist.shape[0], 1) # 180 deg rotation around z
        wrist_rot_quat = quat_mul(z_rot_180, wrist_rot_quat)
        plus_y_rot, minus_x_rot, plus_z_rot = get_euler_xyz(avp_output_wrist[:,3:7].to(self.device))  # Convert quaternion to euler angles

        hand_pose_ref[3] -= minus_x_rot
        hand_pose_ref[4] += (plus_y_rot-1.57)
        hand_pose_ref[5] += (plus_z_rot-4.71) 

        y_rot_mius_90 = torch.tensor([0.7071068, 0, -0.7071068, 0], device=self.device).repeat(self.num_envs, 1) # quaternion for -90 deg y rotation
        z_rot_mius_90 = torch.tensor([0.7071068, 0, 0, -0.7071068], device=self.device).repeat(self.num_envs, 1) # quaternion for -90 deg z rotation

        wrist_rot_quat = quat_mul(wrist_rot_quat,z_rot_mius_90)
        wrist_rot_quat = quat_mul(wrist_rot_quat,y_rot_mius_90)
        rotation_error = orientation_error(wrist_rot_quat, self.right_hand_rot)

        self.rot_error_integral += rotation_error * self.dt
        self.rot_error_integral = torch.clamp(self.rot_error_integral, -1, 1)
        rot_derivative = (rotation_error - self.prev_rot_error) / self.dt

        self.Kp_rot = 40 # 0.3
        self.Ki_rot = 0.001
        self.Kd_rot = 1.5 # 0.005
        #! for urdf with original inertia
        # self.Kp_rot = 1
        # self.Ki_rot = 0.001
        # self.Kd_rot = 0.05
        torque = self.Kp_rot * rotation_error + self.Ki_rot * self.rot_error_integral + self.Kd_rot * rot_derivative

        self.apply_torque[:, 1, :] = torque

        
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        #* Finger Control
        # index middle ring thumb -> index thumb middle ring
        avp_output_joints = [avp_output_joints[1],avp_output_joints[0],avp_output_joints[2],avp_output_joints[3], 
                        avp_output_joints[12],avp_output_joints[13],avp_output_joints[14],avp_output_joints[15], 
                        avp_output_joints[5],avp_output_joints[4],avp_output_joints[6],avp_output_joints[7], 
                        avp_output_joints[9],avp_output_joints[8],avp_output_joints[10],avp_output_joints[11]]
        
        finger_traj_ref = torch.tensor(avp_output_joints, device=self.device,dtype=torch.float).repeat(self.num_envs, 1)

        self.cur_targets[:, :] = finger_traj_ref

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.cur_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
        
        self.prev_pos_error = position_error
        self.prev_rot_error = rotation_error
        self.prev_targets = self.cur_targets  
    
    def xarm6_hand_action(self, actions):
        #! vision_pro_teleop
        avp_output_joints, avp_output_wrist = self.teleop_ik.get_avp_data()

        avp_output_wrist = avp_output_wrist.to(self.device)
        avp_output_wrist[:,0] = -avp_output_wrist[:,0]  # x -> -x
        avp_output_wrist[:,1] = -avp_output_wrist[:,1]  # y -> -y
        wrist_rot_quat = avp_output_wrist[:,3:7].to(self.device)
        z_rot_180 = torch.tensor([0, 0, 1, 0], device=self.device).repeat(avp_output_wrist.shape[0], 1) # 180 deg rotation around z
        wrist_rot_quat = quat_mul(z_rot_180, wrist_rot_quat)

        y_rot_mius_90 = torch.tensor([0.7071068, 0, -0.7071068, 0], device=self.device).repeat(self.num_envs, 1) # quaternion for -90 deg y rotation
        z_rot_mius_90 = torch.tensor([0.7071068, 0, 0, -0.7071068], device=self.device).repeat(self.num_envs, 1) # quaternion for -90 deg z rotation

        wrist_rot_quat = quat_mul(wrist_rot_quat,z_rot_mius_90)
        wrist_rot_quat = quat_mul(wrist_rot_quat,y_rot_mius_90)


        hand_pose_ref = [0,0,0,0,0, 0]
        hand_pose_ref[0:3] = avp_output_wrist.squeeze(0)[:3].to(self.device)

        hand_pose_ref[1] += 0.4
        hand_pose_ref[2] -= 0.85 # seat height for teleop
        plus_y_rot, minus_x_rot, plus_z_rot = get_euler_xyz(avp_output_wrist[:,3:7].to(self.device))  # Convert quaternion to euler angles

        hand_pose_ref[3] -= minus_x_rot
        hand_pose_ref[4] += (plus_y_rot-1.57)
        hand_pose_ref[5] += (plus_z_rot-4.71) 
        right_hand_target_pos = torch.tensor([hand_pose_ref[0], 
                                            hand_pose_ref[1], 
                                            hand_pose_ref[2]+ 0.1153], device=self.device,dtype=torch.float).repeat(self.num_envs, 1)
                                            # 0.7153 is the height of the table + object center height
        # right_hand_target_pos = torch.tensor([0,0,1], device=self.device,dtype=torch.float).repeat(self.num_envs, 1)

        right_pos_err = (right_hand_target_pos - self.right_hand_pos)
        right_target_rot = wrist_rot_quat
        right_rot_err = orientation_error(right_target_rot, self.right_hand_rot)

        #* xarm6 control
        right_dpose = torch.cat([right_pos_err, right_rot_err], -1).unsqueeze(-1).to(torch.float)
        # print(right_dpose.shape,self.jacobian[:, 6, :, :6].shape)
        right_delta = control_ik(
            self.jacobian[:4, 6, :, :6].to(torch.float),
            self.device,
            right_dpose,
            self.num_envs,
        )
        self.cur_targets[:, 0:6] = self.dexterous_hand_dof_pos[:, 0:6] + right_delta[:, :6]   

        # print(self.cur_targets[0,:6])

        #* Finger Control
        # index middle ring thumb -> index thumb middle ring
        avp_output_joints = [avp_output_joints[1],avp_output_joints[0],avp_output_joints[2],avp_output_joints[3], 
                        avp_output_joints[12],avp_output_joints[13],avp_output_joints[14],avp_output_joints[15], 
                        avp_output_joints[5],avp_output_joints[4],avp_output_joints[6],avp_output_joints[7], 
                        avp_output_joints[9],avp_output_joints[8],avp_output_joints[10],avp_output_joints[11]]
        
        finger_traj_ref = torch.tensor(avp_output_joints, device=self.device,dtype=torch.float).repeat(self.num_envs, 1)

        self.cur_targets[:, 6:] = finger_traj_ref

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))



        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.cur_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
    
    def pre_physics_step(self, actions):

        self.actions = actions.clone().to(self.device)

        if self.cfg['env']['use_xarm6']:
            self.xarm6_hand_action(self.actions)
        else:
            # self.armless_hand_action(self.actions)
            self.keyboard_action(self.actions)

        

    def post_physics_step(self):

        self.progress_buf += 1
        # compute observation and reward
        if self.config['Modes']['train_default']: self.compute_observations_default()
        else: self.compute_observations()

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.3)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.3)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.3)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])
    
    def add_debug_vector(self, env, pos, vec, magnitude=1,color=[0.85, 0.1, 0.1]):
        # Scale the vector to the desired magnitude
        scaled_vec = vec * magnitude
        # Add vector to position to get end
        end_pos = (pos + scaled_vec).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], end_pos[0], end_pos[1], end_pos[2]], color)

    # # ---------------------- Compute Reward ---------------------- # #
    def compute_reward(self, actions, id=-1):
        '''
        policy_mode, see dedicated_policy.yaml
        1: end2end_policy
        2: pure_singulation_policy
        3: pure_grasping_policy
        '''

        self.end2end_policy = self.config['Modes']['end2end_policy']
        self.pure_singulation_policy = self.config['Modes']['pure_singulation_policy']
        self.pure_grasping_policy = self.config['Modes']['pure_grasping_policy']

        self.rew_buf[:] = 0
        self.progress_buf[:] = 0
        self.successes[:] = 0
        self.current_successes[:] = 0
        self.consecutive_successes[:] = 0
        self.final_successes[:] = 0

        
        # append successes
        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

    # # ---------------------- Compute Observations ---------------------- # #
    def compute_observations(self):
        # refresh state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)

        wrist_idx = self.hand_body_idx_dict['wrist']
        self.right_hand_pos = self.rigid_body_states[:, wrist_idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, wrist_idx, 3:7]


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def control_ik(j_eef, device: str, dpose, num_envs: int):
    # Set controller parameters
    # IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u