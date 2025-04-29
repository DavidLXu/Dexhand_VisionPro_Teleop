# DexHand Trajectory Replay

This is a simplified script to replay recorded hand and object trajectories in the Isaac Gym simulator. The script removes all the teleoperation and control code, focusing only on replaying pre-recorded trajectories.

## Features

- Loads and replays pre-recorded trajectory files (JSON format)
- Visualizes both hand and object movement without requiring any input devices
- Simplified codebase with only the essential functionality

## Requirements

- Isaac Gym
- PyTorch
- PyTorch3D
- NumPy

## Usage

Run the trajectory replay script with:

```bash
python DexTeleop/dexgrasp/tasks/replay_trajectory.py --trajectory PATH_TO_TRAJECTORY_FILE
```

By default, it will load the trajectory file at `DexTeleop/dexgrasp/recorded_trajectories/trajectory_20250429-190453.json`.

### Command-line Arguments

- `--headless`: Run without visualization (useful for performance testing)
- `--trajectory`: Path to the trajectory file to replay

### Keyboard Controls

When running with visualization:

- `ESC`: Quit the program
- `V`: Toggle viewer sync
- `R`: Start replaying the trajectory (if stopped)
- `T`: Stop replaying the trajectory

## Trajectory File Format

The trajectory file should be a JSON file containing a list of frames, where each frame has:

- `hand_pos`: 3D position of the hand [x, y, z]
- `hand_rot`: Quaternion rotation of the hand [x, y, z, w]
- `finger_pos`: Joint angles for the hand's fingers
- `object_pos`: 3D position of the manipulated object [x, y, z] (optional)
- `object_rot`: Quaternion rotation of the object [x, y, z, w] (optional)

## How It Works

The script:
1. Loads the trajectory file and converts it to PyTorch tensors
2. Creates a minimal Isaac Gym simulation with just the hand and object
3. For each frame in the trajectory:
   - Sets the hand position and orientation
   - Sets the finger joint positions
   - Sets the object position and orientation (if present)
   - Advances the simulation
4. Automatically exits when the trajectory is completed

## Customization

To use a different object model, modify the `object_asset_file` path in the `_create_envs` method. 