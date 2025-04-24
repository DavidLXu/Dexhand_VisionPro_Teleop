import pybullet as p
import numpy as np
import os
import time
from avp_stream import VisionProStreamer
from utils.se3_utils import * 

'''
This is based off of https://github.com/Improbable-AI/VisionProTeleop by Younghyo Park et. al. which streams the AVP data.
We then runs inverse kinematics and output LEAP Hand joint angles.
Feed these joint angles into your preferred LEAP Hand API, Python, ROS1/ROS2 etc.

See https://github.com/leap-hand/Bidex_VisionPro_Teleop for further details.

Note how the fingertip positions are matching, but the joint angles between the two hands are not due to the IK solution.  :) 
This is not as accurate as Manus gloves but easier to use.
Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
'''

#Set this to your IP address of you AVP
class AllegroPybulletIKPython():
    def __init__(self, AVP_IP = "192.168.100.17"):
        self.vps = VisionProStreamer(ip = AVP_IP, record = True)
        p.connect(p.GUI)
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)

        self.leapEndEffectorIndex = [3, 4, 8, 9, 13, 14, 18, 19]

        path_src = os.path.join(path_src, "../../../Assets/allegro_hand/allegro_hand_right.urdf")

        self.AllegroId = p.loadURDF(
            path_src,
            [-0.2,0,0],
            p.getQuaternionFromEuler([1.57, 0, -1.57]),   ##this is the correct frame and rotation for left hand.....use left hand data instead
            # p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase = True
        )

        self.numJoints = p.getNumJoints(self.AllegroId)
        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)
        self.create_target_vis()

                # Set camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=10,
            cameraPitch=-30,
            cameraTargetPosition=[-0.2,0,0]
        )
        self.index = 0


    def create_target_vis(self):
        # load balls
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0, 0, 0]
        
        self.ballMbt = []
        for i in range(0,4):
            self.ballMbt.append(p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)) # for base and finger tip joints    
            no_collision_group = 0
            no_collision_mask = 0
            p.setCollisionFilterGroupMask(self.ballMbt[i], -1, no_collision_group, no_collision_mask)
        p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1]) 
        p.changeVisualShape(self.ballMbt[1], -1, rgbaColor=[0, 1, 0, 1]) 
        p.changeVisualShape(self.ballMbt[2], -1, rgbaColor=[0, 0, 1, 1])  
        p.changeVisualShape(self.ballMbt[3], -1, rgbaColor=[1, 1, 1, 1])
        
    def update_target_vis(self, hand_pos):
        _, current_orientation = p.getBasePositionAndOrientation( self.ballMbt[0])
        p.resetBasePositionAndOrientation(self.ballMbt[0], hand_pos[3], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[1])
        p.resetBasePositionAndOrientation(self.ballMbt[1], hand_pos[5], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[2])
        p.resetBasePositionAndOrientation(self.ballMbt[2], hand_pos[7], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[3])
        p.resetBasePositionAndOrientation(self.ballMbt[3], hand_pos[1], current_orientation)

    def np2tensor(data: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:  
        for key in data.keys():
            data[key] = torch.tensor(data[key], dtype = torch.float32, device = device)
        return data

    def get_avp_data(self):
        #gets the data converts it and then computes IK and visualizes
        r = self.vps.latest              

        hand_pose = np.asarray(r['right_fingers']).astype(float) 


        
        root_pose = mat2posquat(torch.tensor(r['right_wrist']))



        indices = [3,4,8,9,13,14,18,19,23,24]
        hand_pos = hand_pose[indices, :3, 3]

        # thumb
        for i in range(0,2):
            hand_pos[i][0] = hand_pos[i][0] * 1.45 * 1.4
            hand_pos[i][1] = hand_pos[i][1] * 1.4
            hand_pos[i][2] = hand_pos[i][2] * 1.4

        # index
        for i in range(2,4):
            hand_pos[i][0] = hand_pos[i][0] * 1.45 * 1.4
            hand_pos[i][1] = hand_pos[i][1] * 1.4
            hand_pos[i][2] = hand_pos[i][2] * 1.4

        # middle
        for i in range(4,6):
            hand_pos[i][0] = hand_pos[i][0] * 1.45 * 1.4
            hand_pos[i][1] = hand_pos[i][1] * 1.4
            hand_pos[i][2] = hand_pos[i][2] * 1.4
            
        # ring
        for i in range(6,8):
            hand_pos[i][0] = hand_pos[i][0] * 1.45 * 1.4
            hand_pos[i][1] = hand_pos[i][1] * 1.4
            hand_pos[i][2] = hand_pos[i][2] * 1.4


        output = self.compute_IK(hand_pos)
        self.update_target_vis(hand_pos)
        return output, root_pose
        
    def compute_IK(self, hand_pos):
        p.stepSimulation()     

        rightHandIndex_middle_pos = hand_pos[2]
        rightHandIndex_pos = hand_pos[3]
        
        rightHandMiddle_middle_pos = hand_pos[4]
        rightHandMiddle_pos = hand_pos[5]
        
        rightHandRing_middle_pos = hand_pos[6]
        rightHandRing_pos = hand_pos[7]
        
        rightHandThumb_middle_pos = hand_pos[0]
        rightHandThumb_pos = hand_pos[1]
        
        leapEndEffectorPos = [
            rightHandIndex_middle_pos,
            rightHandIndex_pos,
            rightHandMiddle_middle_pos,
            rightHandMiddle_pos,
            rightHandRing_middle_pos,
            rightHandRing_pos,
            rightHandThumb_middle_pos,
            rightHandThumb_pos
        ]

        jointPoses = p.calculateInverseKinematics2(
            self.AllegroId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=0.0001,
        )

        # jointPoses = list(jointPoses)
        # jointPoses = [0.0 for _ in range(16)]

        # print(len(jointPoses))
        # jointPoses[self.index%800//50] = 1.5
        # self.index += 1
        
        combined_jointPoses = (jointPoses[0:4] + (0.0,) + jointPoses[4:8] + (0.0,) + jointPoses[8:12] + (0.0,) + jointPoses[12:16] + (0.0,))
        combined_jointPoses = list(combined_jointPoses)

        # update the hand joints
        for i in range(20):
            p.setJointMotorControl2(
                bodyIndex=self.AllegroId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=combined_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        # map results to real robot
        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        #real_left_robot_hand_q = np.array([0.0 for _ in range(16)])

        real_robot_hand_q[0:4] = jointPoses[0:4]
        real_robot_hand_q[4:8] = jointPoses[4:8]
        real_robot_hand_q[8:12] = jointPoses[8:12]
        real_robot_hand_q[12:16] = jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        return [float(i) for i in real_robot_hand_q]



if __name__ == "__main__":
    # pbik = Leapv1PybulletIKPython(is_left=False)
    pbik = AllegroPybulletIKPython()
    while True:
        output_joints = pbik.get_avp_data()
        # print(output_joints)  ##This is the 16 dimensional LEAP Hand vector  Feed this into your python or ROS1/ROS2 code.  Ordering is the same!
        time.sleep(0.03)