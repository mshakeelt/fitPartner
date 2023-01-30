from . import math3d
from . import bvh_writer
import numpy as np

class Skeleton(object):
    def __init__(self):
        self.root = 'Hips'
        self.keypoint2index = {
            'Nose': 0,
            'Neck': 1,
            'RightShoulder': 2,
            'RightElbow': 3,
            'RightWrist': 4,
            'LeftShoulder': 5,
            'LeftElbow': 6,
            'LeftWrist': 7,
            'Hips': 8,
            'RightHip': 9,
            'RightKnee': 10,
            'RightAnkle': 11,
            'LeftHip': 12,
            'LeftKnee': 13,
            'LeftAnkle': 14,
            'RightEye': 15,
            'LeftEye': 16,
            'RightEarEndSite': 17,
            'LeftEarEndSite': 18,
            'LeftBigToeEndSite': 19,
            'LeftSmallToeEndSite': 20,
            'LeftHeelEndSite': 21,
            'RightBigToeEndSite': 22,
            'RightSmallToeEndSite': 23,
            'RightHeelEndSite': 24,
            'RightPalm': 25,
            'RightThumb1': 26,
            'RightThumb2': 27,
            'RightThumb3': 28,
            'RightThumb4EndSite': 29,
            'RightIndex1': 30,
            'RightIndex2': 31,
            'RightIndex3': 32,
            'RightIndex4EndSite': 33,
            'RightMiddle1': 34,
            'RightMiddle2': 35,
            'RightMiddle3': 36,
            'RightMiddle4EndSite': 37,
            'RightRing1': 38,
            'RightRing2': 39,
            'RightRing3': 40,
            'RightRing4EndSite': 41,
            'RightBaby1': 42,
            'RightBaby2': 43,
            'RightBaby3': 44,
            'RightBaby4EndSite': 45,
            'LeftPalm': 46,
            'LeftThumb1': 47,
            'LeftThumb2': 48,
            'LeftThumb3': 49,
            'LeftThumb4EndSite': 50,
            'LeftIndex1': 51,
            'LeftIndex2': 52,
            'LeftIndex3': 53,
            'LeftIndex4EndSite': 54,
            'LeftMiddle1': 55,
            'LeftMiddle2': 56,
            'LeftMiddle3': 57,
            'LeftMiddle4EndSite': 58,
            'LeftRing1': 59,
            'LeftRing2': 60,
            'LeftRing3': 61,
            'LeftRing4EndSite': 62,
            'LeftBaby1': 63,
            'LeftBaby2': 64,
            'LeftBaby3': 65,
            'LeftBaby4EndSite': 66,
        }
        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)
        self.children = {
            'Hips': ['RightHip', 'LeftHip', 'Neck'],
            'RightHip': ['RightKnee'],
            'RightKnee': ['RightAnkle'],
            'RightAnkle': ['RightBigToeEndSite', 'RightSmallToeEndSite', 'RightHeelEndSite'],
            'RightBigToeEndSite': [],
            'RightSmallToeEndSite': [],
            'RightHeelEndSite': [],
            'LeftHip': ['LeftKnee'],
            'LeftKnee': ['LeftAnkle'],
            'LeftAnkle': ['LeftBigToeEndSite', 'LeftSmallToeEndSite', 'LeftHeelEndSite'],
            'LeftBigToeEndSite': [],
            'LeftSmallToeEndSite': [],
            'LeftHeelEndSite': [],
            'Neck': ['RightShoulder', 'LeftShoulder', 'Nose'],
            'RightShoulder': ['RightElbow'],
            'RightElbow': ['RightWrist'],
            'RightWrist': ['RightPalm'],
            'RightPalm': ['RightThumb1', 'RightIndex1', 'RightMiddle1', 'RightRing1', 'RightBaby1'],
            'RightThumb1': ['RightThumb2'],
            'RightThumb2': ['RightThumb3'],
            'RightThumb3': ['RightThumb4EndSite'],
            'RightThumb4EndSite': [],
            'RightIndex1': ['RightIndex2'],
            'RightIndex2': ['RightIndex3'],
            'RightIndex3': ['RightIndex4EndSite'],
            'RightIndex4EndSite': [],
            'RightMiddle1': ['RightMiddle2'],
            'RightMiddle2': ['RightMiddle3'],
            'RightMiddle3': ['RightMiddle4EndSite'],
            'RightMiddle4EndSite': [],
            'RightRing1': ['RightRing2'],
            'RightRing2': ['RightRing3'],
            'RightRing3': ['RightRing4EndSite'],
            'RightRing4EndSite': [],
            'RightBaby1': ['RightBaby2'],
            'RightBaby2': ['RightBaby3'],
            'RightBaby3': ['RightBaby4EndSite'],
            'RightBaby4EndSite': [],
            'LeftShoulder': ['LeftElbow'],
            'LeftElbow': ['LeftWrist'],
            'LeftWrist': ['LeftPalm'],
            'LeftPalm': ['LeftThumb1', 'LeftIndex1', 'LeftMiddle1', 'LeftRing1', 'LeftBaby1'],
            'LeftThumb1': ['LeftThumb2'],
            'LeftThumb2': ['LeftThumb3'],
            'LeftThumb3': ['LeftThumb4EndSite'],
            'LeftThumb4EndSite': [],
            'LeftIndex1': ['LeftIndex2'],
            'LeftIndex2': ['LeftIndex3'],
            'LeftIndex3': ['LeftIndex4EndSite'],
            'LeftIndex4EndSite': [],
            'LeftMiddle1': ['LeftMiddle2'],
            'LeftMiddle2': ['LeftMiddle3'],
            'LeftMiddle3': ['LeftMiddle4EndSite'],
            'LeftMiddle4EndSite': [],
            'LeftRing1': ['LeftRing2'],
            'LeftRing2': ['LeftRing3'],
            'LeftRing3': ['LeftRing4EndSite'],
            'LeftRing4EndSite': [],
            'LeftBaby1': ['LeftBaby2'],
            'LeftBaby2': ['LeftBaby3'],
            'LeftBaby3': ['LeftBaby4EndSite'],
            'LeftBaby4EndSite': [],
            'Nose': ['RightEye', 'LeftEye'],
            'RightEye': ['RightEarEndSite'],
            'RightEarEndSite': [],
            'LeftEye': ['LeftEarEndSite'],
            'LeftEarEndSite': [],
        }
        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent        
        self.left_joints = [
            joint for joint in self.keypoint2index
            if 'Left' in joint
        ]
        self.right_joints = [
            joint for joint in self.keypoint2index
            if 'Right' in joint
        ]
        # T-pose
        self.initial_directions = {
            'Hips': [0, 0, 0],
            'Nose': [0, 0, 1],
            'Neck': [0, 0, 1],
            'RightShoulder': [-1, 0, 0],
            'RightElbow': [-1, 0, 0],
            'RightWrist': [-1, 0, 0],
            'LeftShoulder': [1, 0, 0],
            'LeftElbow': [1, 0, 0],
            'LeftWrist': [1, 0, 0],
            'RightHip': [-1, 0, 0],
            'RightKnee': [0, 0, -1],
            'RightAnkle': [0, 0, -1],
            'LeftHip': [1, 0, 0],
            'LeftKnee': [0, 0, -1],
            'LeftAnkle': [0, 0, -1],
            'RightEye': [-1, 0, 0],
            'LeftEye': [1, 0, 0],
            'RightEarEndSite': [-1, 0, 0],
            'LeftEarEndSite': [1, 0, 0],
            'LeftBigToeEndSite': [0, 1, 0],
            'LeftSmallToeEndSite': [0, 1, 0],
            'LeftHeelEndSite': [0, -1, 0],
            'RightBigToeEndSite': [0, 1, 0],
            'RightSmallToeEndSite': [0, 1, 0],
            'RightHeelEndSite': [0, -1, 0],
            'RightPalm': [-1, 0, 0],
            'RightThumb1': [-1, 0, 0],
            'RightThumb2': [-1, 0, 0],
            'RightThumb3': [-1, 0, 0],
            'RightThumb4EndSite': [-1, 0, 0],
            'RightIndex1': [-1, 0, 0],
            'RightIndex2': [-1, 0, 0],
            'RightIndex3': [-1, 0, 0],
            'RightIndex4EndSite': [-1, 0, 0],
            'RightMiddle1': [-1, 0, 0],
            'RightMiddle2': [-1, 0, 0],
            'RightMiddle3': [-1, 0, 0],
            'RightMiddle4EndSite': [-1, 0, 0],
            'RightRing1': [-1, 0, 0],
            'RightRing2': [-1, 0, 0],
            'RightRing3': [-1, 0, 0],
            'RightRing4EndSite': [-1, 0, 0],
            'RightBaby1': [-1, 0, 0],
            'RightBaby2': [-1, 0, 0],
            'RightBaby3': [-1, 0, 0],
            'RightBaby4EndSite': [-1, 0, 0],
            'LeftPalm': [1, 0, 0],
            'LeftThumb1': [1, 0, 0],
            'LeftThumb2': [1, 0, 0],
            'LeftThumb3': [1, 0, 0],
            'LeftThumb4EndSite': [1, 0, 0],
            'LeftIndex1': [1, 0, 0],
            'LeftIndex2': [1, 0, 0],
            'LeftIndex3': [1, 0, 0],
            'LeftIndex4EndSite': [1, 0, 0],
            'LeftMiddle1': [1, 0, 0],
            'LeftMiddle2': [1, 0, 0],
            'LeftMiddle3': [1, 0, 0],
            'LeftMiddle4EndSite': [1, 0, 0],
            'LeftRing1': [1, 0, 0],
            'LeftRing2': [1, 0, 0],
            'LeftRing3': [1, 0, 0],
            'LeftRing4EndSite': [1, 0, 0],
            'LeftBaby1': [1, 0, 0],
            'LeftBaby2': [1, 0, 0],
            'LeftBaby3': [1, 0, 0],
            'LeftBaby4EndSite': [1, 0, 0],
        }
        """ self.initial_directions = {
            'Hips': [0, 0, 0],
            'Nose': [1, 0, 1],
            'Neck': [1, 0, 0],
            'RightShoulder': [0, -1, 0],
            'RightElbow': [0, -1, 0],
            'RightWrist': [0, -1, 0],
            'LeftShoulder': [0, 1, 0],
            'LeftElbow': [0, 1, 0],
            'LeftWrist': [0, 1, 0],
            'RightHip': [0, -1, 0],
            'RightKnee': [1, 0, 0],
            'RightAnkle': [1, 0, 0],
            'LeftHip': [0, 1, 0],
            'LeftKnee': [1, 0, 0],
            'LeftAnkle': [1, 0, 0],
            'RightEye': [-1, -1, 0],
            'LeftEye': [-1, 1, 0],
            'RightEarEndSite': [0, -1, 0],
            'LeftEarEndSite': [0, 1, 0],
            'LeftBigToeEndSite': [1, 0, 0],
            'LeftSmallToeEndSite': [1, 0, 0],
            'LeftHeelEndSite': [1, 0, 0],
            'RightBigToeEndSite': [1, 0, 0],
            'RightSmallToeEndSite': [1, 0, 0],
            'RightHeelEndSite': [1, 0, 0],
            'RightPalm': [0, -1, 0],
            'RightThumb1': [0, -1, 0],
            'RightThumb2': [0, -1, 0],
            'RightThumb3': [0, -1, 0],
            'RightThumb4EndSite': [0, -1, 0],
            'RightIndex1': [0, -1, 0],
            'RightIndex2': [0, -1, 0],
            'RightIndex3': [0, -1, 0],
            'RightIndex4EndSite': [0, -1, 0],
            'RightMiddle1': [0, -1, 0],
            'RightMiddle2': [0, -1, 0],
            'RightMiddle3': [0, -1, 0],
            'RightMiddle4EndSite': [0, -1, 0],
            'RightRing1': [0, -1, 0],
            'RightRing2': [0, -1, 0],
            'RightRing3': [0, -1, 0],
            'RightRing4EndSite': [0, -1, 0],
            'RightBaby1': [0, -1, 0],
            'RightBaby2': [0, -1, 0],
            'RightBaby3': [0, -1, 0],
            'RightBaby4EndSite': [0, -1, 0],
            'LeftPalm': [0, 1, 0],
            'LeftThumb1': [0, 1, 0],
            'LeftThumb2': [0, 1, 0],
            'LeftThumb3': [0, 1, 0],
            'LeftThumb4EndSite': [0, 1, 0],
            'LeftIndex1': [0, 1, 0],
            'LeftIndex2': [0, 1, 0],
            'LeftIndex3': [0, 1, 0],
            'LeftIndex4EndSite': [0, 1, 0],
            'LeftMiddle1': [0, 1, 0],
            'LeftMiddle2': [0, 1, 0],
            'LeftMiddle3': [0, 1, 0],
            'LeftMiddle4EndSite': [0, 1, 0],
            'LeftRing1': [0, 1, 0],
            'LeftRing2': [0, 1, 0],
            'LeftRing3': [0, 1, 0],
            'LeftRing4EndSite': [0, 1, 0],
            'LeftBaby1': [0, 1, 0],
            'LeftBaby2': [0, 1, 0],
            'LeftBaby3': [0, 1, 0],
            'LeftBaby4EndSite': [0, 1, 0],
        } """

    def get_initial_offset(self, poses_3d):
        # TODO: RANSAC
        bone_lens = {self.root: [0]}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            p_idx = self.keypoint2index[parent]
            p_name = parent
            while p_idx == -1:
                # find real parent
                p_name = self.parent[p_name]
                p_idx = self.keypoint2index[p_name]
            for child in self.children[parent]:
                stack.append(child)

                if self.keypoint2index[child] == -1:
                    bone_lens[child] = [0.1]
                else:
                    c_idx = self.keypoint2index[child]
                    bone_lens[child] = np.linalg.norm(
                        poses_3d[:, p_idx] - poses_3d[:, c_idx],
                        axis=1
                    )
        bone_len = {}
        for joint in self.keypoint2index:
            if 'Left' in joint or 'Right' in joint:
                base_name = joint.replace('Left', '').replace('Right', '')
                left_len = np.mean(bone_lens['Left' + base_name])
                right_len = np.mean(bone_lens['Right' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])
        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]
        return initial_offset

    def get_bvh_header(self, poses_3d):
        initial_offset = self.get_initial_offset(poses_3d)
        nodes = {}
        for joint in self.keypoint2index:
            is_root = joint == self.root
            is_end_site = 'EndSite' in joint
            nodes[joint] = bvh_writer.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',
                is_root=is_root,
                is_end_site=is_end_site,
            )
        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]
        header = bvh_writer.BvhHeader(root=nodes[self.root], nodes=nodes)
        return header

    def pose2euler(self, pose, header):
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]    
            if node.is_root:
                channel.extend(pose[joint_idx])
            index = self.keypoint2index
            order = None
            if joint == 'Hips':
                x_dir = pose[index['LeftHip']] - pose[index['RightHip']]
                y_dir = None
                z_dir = pose[index['Neck']] - pose[joint_idx]
                order = 'zyx'
            elif joint in ['RightHip', 'RightKnee']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['Hips']] - pose[index['RightHip']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'
            elif joint in ['LeftHip', 'LeftKnee']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['LeftHip']] - pose[index['Hips']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'
            elif joint == 'Neck':
                x_dir = None
                y_dir = pose[index['Nose']] - pose[joint_idx]
                z_dir = pose[joint_idx] - pose[index['Hips']]
                order = 'zxy'
            elif joint == 'LeftShoulder':
                x_dir = pose[index['LeftElbow']] - pose[joint_idx]
                y_dir = pose[index['LeftElbow']] - pose[index['LeftWrist']]
                z_dir = None
                order = 'xzy'
            elif joint == 'LeftElbow':
                x_dir = pose[index['LeftWrist']] - pose[joint_idx]
                y_dir = pose[joint_idx] - pose[index['LeftShoulder']]
                z_dir = None
                order = 'xzy'
            elif joint == 'RightShoulder':
                x_dir = pose[joint_idx] - pose[index['RightElbow']]
                y_dir = pose[index['RightElbow']] - pose[index['RightWrist']]
                z_dir = None
                order = 'xzy'
            elif joint == 'RightElbow':
                x_dir = pose[joint_idx] - pose[index['RightWrist']]
                y_dir = pose[joint_idx] - pose[index['RightShoulder']]
                z_dir = None
                order = 'xzy'    
            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()    
            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )    
            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)
            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)
        return channel

    def poses2bvh(self, poses_3d, header=None, output_file=None):
        if not header:
            header = self.get_bvh_header(poses_3d)
        channels = []
        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))
        if output_file:
            bvh_writer.write_bvh(output_file, header, channels)
        return channels, header