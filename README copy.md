# DexHand Vision Pro Teleoperation

A versatile teleoperation system for dexterous robotic hands, supporting multiple input devices including Apple Vision Pro, SpaceMouse, keyboard, and joystick.

## Features

- **Multi-Device Support**
  - Apple Vision Pro integration for natural hand tracking
  - SpaceMouse 3D controller support
  - Keyboard teleoperation
  - Joystick control
  - Extensible device interface for adding new input methods

- **Robotic Hand Control**
  - Dexterous hand manipulation
  - Position and orientation control
  - Finger joint control
  - Real-time physics simulation

- **Advanced Features**
  - Trajectory recording and replay
  - Debug visualization
  - Multiple environment support
  - Flexible configuration system

## Requirements

- Python 3.8+
- PyTorch
- IsaacGym
- PyBullet
- PyGame
- PySpaceMouse
- Apple Vision Pro SDK (for Vision Pro support)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DexHand_VisionPro_Teleop.git
cd DexHand_VisionPro_Teleop
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up IsaacGym:
```bash
# Follow IsaacGym installation instructions
# https://developer.nvidia.com/isaac-gym
```

## Usage

### Configuration

The system is configured through YAML files. Example configuration:

```yaml
env:
  teleop_device: "vision_pro"  # or "keyboard", "spacemouse", "joystick"
  vision_pro_ip: "192.168.100.17"  # Vision Pro IP address
  num_envs: 1  # Number of parallel environments
```

### Running the Teleoperation

1. Start the simulation:
```bash
python run_teleop.py --config configs/dexhand_teleop.yaml
```

2. Control options:
- **Vision Pro**: Natural hand tracking
- **SpaceMouse**: 3D controller for position and orientation
- **Keyboard**:
  - WASD: Position control
  - IJKLUO: Rotation control
  - F: Grasp
  - H: Release
- **Joystick**: Analog control for position and orientation

### Recording and Playback

- Start recording: Press 'R'
- Stop recording: Press 'S'
- Start playback: Press 'P'
- Stop playback: Press 'Q'

## Project Structure

```
DexHand_VisionPro_Teleop/
├── DexTeleop/              # Main teleoperation code
│   ├── dexgrasp/          # Grasping algorithms
│   └── tasks/             # Task definitions
├── Assets/                # 3D models and assets
├── YCB/                   # YCB object models
├── Logs/                  # Recording and playback data
└── Media/                 # Media files
```

## Extending the System

### Adding New Input Devices

1. Create a new class inheriting from `TeleopDevice`:
```python
class NewDevice(TeleopDevice):
    def __init__(self, device: str, num_envs: int):
        super().__init__("new_device", device, num_envs)
        # Initialize your device
        
    def get_teleop_data(self):
        # Return position, rotation, and finger values
        return target_pos, target_quat, finger_val
```

2. Add device initialization in `DexhandTeleop`:
```python
if cfg['env'].get('teleop_device') == 'new_device':
    self.teleop_device = NewDevice(device=self.device, num_envs=self.num_envs)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the NVIDIA License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA IsaacGym team
- Apple Vision Pro team
- 3DConnexion for SpaceMouse support
