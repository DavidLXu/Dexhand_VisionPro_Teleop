import numpy as np
import torch

import pygame
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
from isaacgym import gymapi

class VisionProTeleopDevice(TeleopDevice):
    def __init__(self, ip_address: str, device: str):
        super().__init__("vision_pro", device)
        from utils.avp_leap import AllegroPybulletIKPython
        # Use DIRECT mode instead of GUI to avoid multiple GUI connections
        self.teleop_ik = AllegroPybulletIKPython(ip_address, gui=False)
