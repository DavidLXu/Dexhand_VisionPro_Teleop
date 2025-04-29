#!/usr/bin/env python3

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import json

import argparse
import sys

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_apply, to_torch
import torch
from hand_base.base_task import BaseTask
from typing import Union, Tuple
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)

class Compose:
    def __init__(self, transforms: list):
        """Composes several transforms together."""
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

def orientation_error(desired, current):
    # Compute orientation error
    quat_diff = quat_mul(desired, quat_conjugate(current))
    return quat_diff[:, 1:4] * torch.sign(quat_diff[:, 0]).unsqueeze(-1)

def quat_mul(a, b):
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1)

def quat_conjugate(a):
    return torch.stack([-a[:, 0], -a[:, 1], -a[:, 2], a[:, 3]], dim=-1)

class TrajectoryReplayer:
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, trajectory_file, gym):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless
        self.trajectory_file = trajectory_file
        self.gym = gym  # Pass the gym directly
        
        # Determine device
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda:" + str(self.device_id)
        else:
            self.device = "cpu"

        # Initialize the sim
        self.sim = None
        self.viewer = None
        self.num_envs = 1  # Simplified to just one environment
        self.enable_viewer_sync = True

        # Initialize flags
        self.is_replaying = False
        self.no_dynamics_replay = True  # Always use no dynamics mode
        self.replay_trajectory = None
        self.replay_index = 0
        self.exit_after_replay = True

        # Create the simulation
        self.create_sim()
        
        # Set initial states
        self.envs = []
        self.dexterous_hand_actor = None
        self.object_actor = None
        self.init_data()
        
        # Create the environment
        self._create_ground_plane()
        self._create_envs(self.num_envs, 1.0, 1)
        
        # Set up viewer
        if not self.headless:
            # Create viewer
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1920
            camera_props.height = 1080
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
                
            # Set camera position and target
            cam_pos = gymapi.Vec3(3.0, 0.0, 2.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
            # Add keyboard controls
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "start_replaying")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "stop_replaying")
            
        # Get relevant tensor handles
        self.get_tensor_handles()
        
        # Load the trajectory immediately
        self.load_trajectory(self.trajectory_file)
        self.start_replay()

    def init_data(self):
        # Initialize data structures needed for replay
        self.progress_buf = None
        self.obs_buf = None
        
    def create_sim(self):
        # Create the Isaac Gym simulation
        self.sim = self.gym.create_sim(
            self.device_id, self.device_id, 
            self.physics_engine, 
            self.sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
            
    def _create_ground_plane(self):
        # Create a simple ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_envs(self, num_envs, spacing, num_per_row):
        # Simple environment creation with just the hand
        asset_root = "../../Assets"
        dexterous_hand_asset_file = "allegro_hand/allegro_hand_right_glb.urdf"
        object_asset_root = "../../"
        object_asset_file = "YCB/021_bleach_cleanser/scaled_textured_simple.urdf"
        
        print(f"Loading hand asset from: {asset_root}/{dexterous_hand_asset_file}")
        print(f"Loading object asset from: {asset_root}/{object_asset_file}")
        
        # Load assets
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
            
        # Load the hand asset
        self.dexterous_hand_asset = self.gym.load_asset(
            self.sim, asset_root, dexterous_hand_asset_file, asset_options)
            
        # Output the number of bodies and DOFs in the hand asset
        num_bodies = self.gym.get_asset_rigid_body_count(self.dexterous_hand_asset)
        num_dofs = self.gym.get_asset_dof_count(self.dexterous_hand_asset)
        print(f"Hand asset has {num_bodies} rigid bodies and {num_dofs} DOFs")
        
        # List all the body names
        body_names = [self.gym.get_asset_rigid_body_name(self.dexterous_hand_asset, i) for i in range(num_bodies)]
        print(f"Hand body names: {body_names}")
        
        # List all the DOF names
        dof_names = [self.gym.get_asset_dof_name(self.dexterous_hand_asset, i) for i in range(num_dofs)]
        print(f"Hand DOF names: {dof_names}")
            
        # Load the object asset
        self.object_asset = self.gym.load_asset(
            self.sim, object_asset_root, object_asset_file, asset_options)
            
        # Setup environments
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # Define start pose for the hand
        hand_start_pose = gymapi.Transform()
        hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.2)  # Lower height for Allegro hand
        hand_start_pose.r = gymapi.Quat(0.5, 0.5, 0.5, 0.5)  # Start with palm facing forward
        
        # Define start pose for the object
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.05)  # Lower position for object
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        # Create environment
        env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
        self.envs.append(env)
        
        # Add hand to the environment
        self.dexterous_hand_actor = self.gym.create_actor(env, self.dexterous_hand_asset, hand_start_pose, "hand", 0, 0)
        
        # Get DOF properties and count for the hand
        self.num_dofs = self.gym.get_asset_dof_count(self.dexterous_hand_asset)
        
        # Set DOF properties
        dof_props = self.gym.get_actor_dof_properties(env, self.dexterous_hand_actor)
        for i in range(self.num_dofs):
            dof_props['stiffness'][i] = 100.0
            dof_props['damping'][i] = 40.0
            dof_props['effort'][i] = 100.0
        self.gym.set_actor_dof_properties(env, self.dexterous_hand_actor, dof_props)
        
        # Add object to the environment
        if self.cfg.get('env', {}).get('use_object', True):
            self.object_actor = self.gym.create_actor(env, self.object_asset, object_start_pose, "object", 0, 0)
            
        # Get body indices
        self.hand_body_idx_dict = {}
        # Try to find the wrist body - it might be named differently
        for name in ['wrist', 'palm', 'base_link']:
            idx = self.gym.find_actor_rigid_body_index(env, self.dexterous_hand_actor, name, gymapi.DOMAIN_SIM)
            if idx != -1:
                self.hand_body_idx_dict['wrist'] = idx
                print(f"Found hand wrist/root at body '{name}' with index {idx}")
                break
        
        if 'wrist' not in self.hand_body_idx_dict:
            print("WARNING: Could not find wrist body in hand model, using index 0")
            self.hand_body_idx_dict['wrist'] = 0
        
    def get_tensor_handles(self):
        # Acquire handles to relevant tensors
        self.gym.prepare_sim(self.sim)
        
        # Get root state tensor
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(_actor_root_state_tensor)
        
        # Get rigid body state tensor
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(_rigid_body_state_tensor)
        
        # Get DOF state tensor
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(_dof_state_tensor)
        
        # Initialize progress buffer
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
    def load_trajectory(self, filename):
        """Load a recorded trajectory from file"""
        print(f"Loading trajectory from {filename}")
        with open(filename, 'r') as f:
            trajectory_data = json.load(f)
        
        # Convert lists back to tensors
        self.replay_trajectory = []
        for frame in trajectory_data:
            tensor_frame = {
                'hand_pos': torch.tensor(frame['hand_pos'], device=self.device),
                'hand_rot': torch.tensor(frame['hand_rot'], device=self.device),
                'finger_pos': torch.tensor(frame['finger_pos'], device=self.device)
            }
            if 'object_pos' in frame:
                tensor_frame['object_pos'] = torch.tensor(frame['object_pos'], device=self.device)
                tensor_frame['object_rot'] = torch.tensor(frame['object_rot'], device=self.device)
            self.replay_trajectory.append(tensor_frame)
        
        print(f"Loaded trajectory with {len(self.replay_trajectory)} frames from {filename}")
    
    def start_replay(self):
        """Start replaying a recorded trajectory"""
        self.is_replaying = True
        self.replay_index = 0
        print("Starting replay with no dynamics...")
    
    def stop_replay(self):
        """Stop replaying the trajectory"""
        self.is_replaying = False
        self.replay_trajectory = None
        self.replay_index = 0
        print("Replay stopped")
    
    def replay_step(self):
        """Execute one step of trajectory replay without dynamics"""
        if not self.is_replaying or self.replay_trajectory is None:
            return
            
        if self.replay_index >= len(self.replay_trajectory):
            self.stop_replay()
            # Exit the program if we're in replay_mode and have finished replaying
            if self.exit_after_replay:
                print("Replay completed. Exiting program.")
                return True
            return False
            
        frame = self.replay_trajectory[self.replay_index]
        
        # Get the hand position and orientation from the frame
        hand_pos = frame['hand_pos']
        hand_rot = frame['hand_rot']
        finger_pos = frame['finger_pos']
        
        # Print debugging info for the first frame
        if self.replay_index == 0:
            print(f"Frame 0 - Hand position: {hand_pos}")
            print(f"Frame 0 - Hand rotation: {hand_rot}")
            print(f"Frame 0 - Finger positions shape: {finger_pos.shape}")
            print(f"Rigid body states shape: {self.rigid_body_states.shape}")
            print(f"DOF state shape: {self.dof_state.shape}")
            print(f"Number of DOFs: {self.num_dofs}")
            print(f"Root state tensor shape: {self.root_state_tensor.shape}")
        
        # Find the base link (wrist) index
        wrist_idx = self.hand_body_idx_dict['wrist']
        
        # Update root state of the hand actor (index 0)
        hand_idx = 0  # Hand is the first actor (index 0)
        
        # Update the root state tensor for the hand
        self.root_state_tensor[hand_idx, 0:3] = hand_pos
        self.root_state_tensor[hand_idx, 3:7] = hand_rot
        
        # If there's an object in the scene, set its pose too
        if 'object_pos' in frame and self.object_actor is not None:
            obj_pos = frame['object_pos']
            obj_rot = frame['object_rot']
            
            # Get object index (should be 1)
            obj_idx = 1  # Assuming object is the second actor (index 1)
            
            # Update the root state tensor for the object
            self.root_state_tensor[obj_idx, 0:3] = obj_pos.reshape(3)
            self.root_state_tensor[obj_idx, 3:7] = obj_rot.reshape(4)
        
        # Move root state tensor to CPU before passing to IsaacGym API
        cpu_root_state_tensor = self.root_state_tensor.cpu()
        
        # Apply the updated root state tensor
        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(cpu_root_state_tensor)
        )
        
        # Set the DOF positions for the fingers
        # Reshape and pad/truncate finger positions to match the number of DOFs
        if self.num_dofs > 0:
            finger_pos_padded = torch.zeros(self.num_dofs, device=self.device)
            finger_pos_count = min(len(finger_pos), self.num_dofs)
            finger_pos_padded[:finger_pos_count] = finger_pos[:finger_pos_count]
            
            # Create full DOF state tensor (position and velocity)
            dof_state = torch.zeros(self.num_dofs, 2, device=self.device)
            dof_state[:, 0] = finger_pos_padded  # positions
            # velocities remain zero
            
            # Move to CPU before passing to IsaacGym API
            cpu_dof_state = dof_state.cpu()
            cpu_finger_pos_padded = finger_pos_padded.cpu()
            
            # Set the DOF state
            self.gym.set_dof_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(cpu_dof_state.flatten())
            )
            
            # Also set position targets
            self.gym.set_dof_position_target_tensor(
                self.sim,
                gymtorch.unwrap_tensor(cpu_finger_pos_padded)
            )
        
        self.replay_index += 1
        return False
    
    def step(self, actions=None):
        # Handle viewer events
        if self.viewer is not None:
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    return True
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "start_replaying" and evt.value > 0:
                    self.start_replay()
                elif evt.action == "stop_replaying" and evt.value > 0:
                    self.stop_replay()
        
        # Execute replay step
        if self.is_replaying:
            should_exit = self.replay_step()
            if should_exit:
                return True
                
            if self.replay_index % 10 == 0:
                print(f"Replaying frame {self.replay_index} of {len(self.replay_trajectory)}")
                
        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Update the viewer
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            if self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
        
        # Add a small delay for smoother playback
        if self.is_replaying:
            import time
            time.sleep(0.03)  # Adjust this value to control playback speed
        
        self.progress_buf += 1
        return False

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Replay a trajectory file with hand and object')
    parser.add_argument('--headless', action='store_true', default=False, help='Run in headless mode')
    parser.add_argument('--trajectory', type=str, default='DexTeleop/dexgrasp/recorded_trajectories/trajectory_20250429-190021.json', 
                        help='Path to the trajectory file')
    args = parser.parse_args()
    
    # Create sim params
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    
    # Set physics engine parameters
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.physx.bounce_threshold_velocity = 0.2
    sim_params.physx.max_depenetration_velocity = 100.0
    
    # Basic configuration
    cfg = {
        'env': {
            'use_object': True,
        }
    }
    
    # Initialize the gym
    gym = gymapi.acquire_gym()
    
    # Create the replayer
    replayer = TrajectoryReplayer(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        device_type="cuda",
        device_id=0,
        headless=args.headless,
        trajectory_file=args.trajectory,
        gym=gym  # Pass the gym instance
    )
    
    # Main loop
    running = True
    while running:
        should_exit = replayer.step()
        if should_exit:
            running = False
            
    print("Exiting")
    
    # Cleanup
    if replayer.viewer is not None:
        replayer.gym.destroy_viewer(replayer.viewer)
    replayer.gym.destroy_sim(replayer.sim)

if __name__ == "__main__":
    main() 