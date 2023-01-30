import torch
import cv2
import torchvision.transforms as transforms
import PIL.Image
import os, sys
from pathlib import Path
import json
from statistics import mean
import numpy as np
from scipy.spatial.distance import euclidean
import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects
from backends.trt_pose.foot_pose.modules.keypoints import extract_keypoints, group_keypoints
from backends.trt_pose.foot_pose.modules.load_state import load_state
from backends.trt_pose.foot_pose.modules.pose import Pose
from backends.trt_pose.foot_pose.modules.val import normalize, pad_width
from backends.trt_pose.foot_pose.modules.with_mobilenet import PoseEstimationWithMobileNet
import mediapipe as mp

dir_path = str(Path(__file__).parent.parent)
sys.path.append(dir_path + r'\backends\open_pose\build\python\openpose\Release')
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + r'\backends\open_pose\build\x64\Release;' +  dir_path + r'\backends\open_pose\build\bin;'
import pyopenpose as op

class Mediapipe_Infer_Full(object):
    def __init__(self) -> None:
        self.imagewidth = None
        self.imageheight = None
        self.body_keypoint_names = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye',
                                    'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
                                    'right_shoulder',  'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_baby_finger_4',
                                    'right_baby_finger_4', 'left_index_finger_4', 'right_index_finger_4', 'left_thumb_4', 'right_thumb_4',
                                    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
                                    'right_heel', 'left_big_toe', 'right_big_toe']
        self.right_hand_keypoints_names = ['right_wrist', 'right_thumb_4', 'right_thumb_3', 'right_thumb_2', 'right_thumb',
                                           'right_index_finger_4', 'right_index_finger_3', 'right_index_finger_2', 'right_index_finger',
                                           'right_middle_finger_4', 'right_middle_finger_3', 'right_middle_finger_2', 'right_middle_finger',
                                           'right_ring_finger_4', 'right_ring_finger_3', 'right_ring_finger_2', 'right_ring_finger',
                                           'right_baby_finger_4', 'right_baby_finger_3', 'right_baby_finger_2', 'right_baby_finger']
        self.left_hand_keypoints_names = ['left_wrist', 'left_thumb_4', 'left_thumb_3', 'left_thumb_2', 'left_thumb',
                                          'left_index_finger_4', 'left_index_finger_3', 'left_index_finger_2', 'left_index_finger',
                                          'left_middle_finger_4', 'left_middle_finger_3', 'left_middle_finger_2', 'left_middle_finger',
                                          'left_ring_finger_4', 'left_ring_finger_3', 'left_ring_finger_2', 'left_ring_finger',
                                          'left_baby_finger_4', 'left_baby_finger_3', 'left_baby_finger_2', 'left_baby_finger']
        missing_keypoints = ['left_small_toe', 'right_small_toe']
        self.unknown_keypoints = dict.fromkeys(missing_keypoints)
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        self.mp_pose_image = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
        self.mp_hands_image = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    def execute(self, image):
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_pose_image.process(RGB_image)
        if results.pose_landmarks is not None:
            landmark_list = results.pose_landmarks.landmark
            keypoint_cordinates = [(land_mark.x * self.imagewidth, land_mark.y * self.imageheight)
                                   if 0 <= land_mark.x <= 1 and 0 <= land_mark.y <= 1 else None for land_mark in landmark_list]
            known_keypoints = dict(zip(self.body_keypoint_names, keypoint_cordinates))
            if known_keypoints['left_shoulder'] is not None and known_keypoints['right_shoulder'] is not None:
                known_keypoints['neck'] = ((known_keypoints['left_shoulder'][0] + known_keypoints['right_shoulder'][0])/2,
                                           (known_keypoints['left_shoulder'][1] + known_keypoints['right_shoulder'][1])/2)
            else:
                known_keypoints['neck'] = None
            if known_keypoints['left_hip'] is not None and known_keypoints['right_hip'] is not None:
                known_keypoints['MidHip'] = ((known_keypoints['left_hip'][0] + known_keypoints['right_hip'][0])/2,
                                             (known_keypoints['left_hip'][1] + known_keypoints['right_hip'][1])/2)
            else:
                known_keypoints['MidHip'] = None
        else:
            known_keypoints = dict.fromkeys(self.body_keypoint_names)
            known_keypoints['neck'] = None
            known_keypoints['MidHip'] = None
        hand_keypoints = self.get_hands(RGB_image)
        body_keypoints = {**known_keypoints, **self.unknown_keypoints}
        full_body_keypoints = self.remove_duplicates_and_join(body_keypoints, hand_keypoints)
        return full_body_keypoints

    def get_hands(self, imageRGB):
        results = self.mp_hands_image.process(imageRGB)
        if results.multi_handedness is None:
            right_hand = dict.fromkeys(self.right_hand_keypoints_names)
            left_hand = dict.fromkeys(self.left_hand_keypoints_names)
        elif len(results.multi_handedness) == 1:
            multi_handedness_0 = results.multi_handedness[0].classification
            label = multi_handedness_0[0].label
            landmark_list = results.multi_hand_landmarks[0].landmark
            keypoint_cordinates = [(land_mark.x * self.imagewidth, land_mark.y * self.imageheight) if 0 <= land_mark.x <= 1 and 0 <= land_mark.y <= 1 else None for land_mark in landmark_list]
            if label == 'Right':
                right_hand = dict.fromkeys(self.right_hand_keypoints_names)
                left_hand = dict(zip(self.left_hand_keypoints_names, keypoint_cordinates))
            else:
                right_hand = dict(zip(self.right_hand_keypoints_names, keypoint_cordinates))
                left_hand = dict.fromkeys(self.left_hand_keypoints_names)
        else:
            multi_handedness_0 = results.multi_handedness[0].classification
            index_one = multi_handedness_0[0].index
            score_one = multi_handedness_0[0].score
            label_one = multi_handedness_0[0].label
            multi_handedness_1 = results.multi_handedness[1].classification
            index_two = multi_handedness_1[0].index
            score_two = multi_handedness_1[0].score
            label_two = multi_handedness_1[0].label
            landmark_list_one = results.multi_hand_landmarks[0].landmark
            landmark_list_two = results.multi_hand_landmarks[1].landmark
            keypoint_cordinates_one = [(land_mark.x * self.imagewidth, land_mark.y * self.imageheight) if 0 <= land_mark.x <= 1 and 0 <= land_mark.y <= 1 else None for land_mark in landmark_list_one]
            keypoint_cordinates_two = [(land_mark.x * self.imagewidth, land_mark.y * self.imageheight) if 0 <= land_mark.x <= 1 and 0 <= land_mark.y <= 1 else None for land_mark in landmark_list_two]
            if label_one != label_two:
                if index_one == 0 and label_one == 'Right':
                    right_hand_cordinates = keypoint_cordinates_two
                    left_hand_cordinates = keypoint_cordinates_one
                elif index_one == 0 and label_one == 'Left':
                    right_hand_cordinates = keypoint_cordinates_one
                    left_hand_cordinates = keypoint_cordinates_two
                elif index_one == 1 and label_one == 'Right':
                    right_hand_cordinates = keypoint_cordinates_one
                    left_hand_cordinates = keypoint_cordinates_two
                elif index_one == 1 and label_one == 'Left':
                    right_hand_cordinates = keypoint_cordinates_two
                    left_hand_cordinates = keypoint_cordinates_one
            else:
                if score_one >= score_two:
                    if index_one == 0:
                        if label_one == 'Right':
                            right_hand_cordinates = None
                            left_hand_cordinates = keypoint_cordinates_one
                        elif label_one == 'Left':
                            right_hand_cordinates = keypoint_cordinates_one
                            left_hand_cordinates = None
                    elif index_one == 1:
                        if label_one == 'Right':
                            right_hand_cordinates = None
                            left_hand_cordinates = keypoint_cordinates_two
                        elif label_one == 'Left':
                            right_hand_cordinates = keypoint_cordinates_two
                            left_hand_cordinates = None
                elif score_one < score_two:
                    if index_two == 0:
                        if label_two == 'Right':
                            right_hand_cordinates = None
                            left_hand_cordinates = keypoint_cordinates_two
                        elif label_two == 'Left':
                            right_hand_cordinates = keypoint_cordinates_two
                            left_hand_cordinates = None
                    elif index_two == 1:
                        if label_two == 'Right':
                            right_hand_cordinates = None
                            left_hand_cordinates = keypoint_cordinates_two
                        elif label_two == 'Left':
                            right_hand_cordinates = keypoint_cordinates_two
                            left_hand_cordinates = None
            if right_hand_cordinates is None:
                right_hand = dict.fromkeys(self.right_hand_keypoints_names)
            else:
                right_hand = dict(zip(self.right_hand_keypoints_names, right_hand_cordinates))
            if left_hand_cordinates is None:
                left_hand = dict.fromkeys(self.left_hand_keypoints_names)
            else:
                left_hand = dict(zip(self.left_hand_keypoints_names, left_hand_cordinates))
        hand_keypoints = {**right_hand, **left_hand}
        return hand_keypoints

    def remove_duplicates_and_join(self, body_keypoints, hand_keypoints):
        if body_keypoints['right_wrist'] is not None and hand_keypoints['right_wrist'] is not None:
            hand_keypoints.pop('right_wrist')
        elif body_keypoints['right_wrist'] is not None and hand_keypoints['right_wrist'] is None:
            hand_keypoints.pop('right_wrist')
        elif body_keypoints['right_wrist'] is None and hand_keypoints['right_wrist'] is not None:
            body_keypoints.pop('right_wrist')
        else:
            hand_keypoints.pop('right_wrist')
        if body_keypoints['left_wrist'] is not None and hand_keypoints['left_wrist'] is not None:
            hand_keypoints.pop('left_wrist')
        elif body_keypoints['left_wrist'] is not None and hand_keypoints['left_wrist'] is None:
            hand_keypoints.pop('left_wrist')
        elif body_keypoints['left_wrist'] is None and hand_keypoints['left_wrist'] is not None:
            body_keypoints.pop('left_wrist')
        else:
            hand_keypoints.pop('left_wrist')
        if body_keypoints['right_baby_finger_4'] is not None and hand_keypoints['right_baby_finger_4'] is not None:
            body_keypoints.pop('right_baby_finger_4')
        elif body_keypoints['right_baby_finger_4'] is not None and hand_keypoints['right_baby_finger_4'] is None:
            hand_keypoints.pop('right_baby_finger_4')
        elif body_keypoints['right_baby_finger_4'] is None and hand_keypoints['right_baby_finger_4'] is not None:
            body_keypoints.pop('right_baby_finger_4')
        else:
            hand_keypoints.pop('right_baby_finger_4')
        if body_keypoints['left_baby_finger_4'] is not None and hand_keypoints['left_baby_finger_4'] is not None:
            body_keypoints.pop('left_baby_finger_4')
        elif body_keypoints['left_baby_finger_4'] is not None and hand_keypoints['left_baby_finger_4'] is None:
            hand_keypoints.pop('left_baby_finger_4')
        elif body_keypoints['left_baby_finger_4'] is None and hand_keypoints['left_baby_finger_4'] is not None:
            body_keypoints.pop('left_baby_finger_4')
        else:
            hand_keypoints.pop('left_baby_finger_4')
        if body_keypoints['right_index_finger_4'] is not None and hand_keypoints['right_index_finger_4'] is not None:
            body_keypoints.pop('right_index_finger_4')
        elif body_keypoints['right_index_finger_4'] is not None and hand_keypoints['right_index_finger_4'] is None:
            hand_keypoints.pop('right_index_finger_4')
        elif body_keypoints['right_index_finger_4'] is None and hand_keypoints['right_index_finger_4'] is not None:
            body_keypoints.pop('right_index_finger_4')
        else:
            hand_keypoints.pop('right_index_finger_4')
        if body_keypoints['left_index_finger_4'] is not None and hand_keypoints['left_index_finger_4'] is not None:
            body_keypoints.pop('left_index_finger_4')
        elif body_keypoints['left_index_finger_4'] is not None and hand_keypoints['left_index_finger_4'] is None:
            hand_keypoints.pop('left_index_finger_4')
        elif body_keypoints['left_index_finger_4'] is None and hand_keypoints['left_index_finger_4'] is not None:
            body_keypoints.pop('left_index_finger_4')
        else:
            hand_keypoints.pop('left_index_finger_4')
        if body_keypoints['right_thumb_4'] is not None and hand_keypoints['right_thumb_4'] is not None:
            body_keypoints.pop('right_thumb_4')
        elif body_keypoints['right_thumb_4'] is not None and hand_keypoints['right_thumb_4'] is None:
            hand_keypoints.pop('right_thumb_4')
        elif body_keypoints['right_thumb_4'] is None and hand_keypoints['right_thumb_4'] is not None:
            body_keypoints.pop('right_thumb_4')
        else:
            hand_keypoints.pop('right_thumb_4')
        if body_keypoints['left_thumb_4'] is not None and hand_keypoints['left_thumb_4'] is not None:
            body_keypoints.pop('left_thumb_4')
        elif body_keypoints['left_thumb_4'] is not None and hand_keypoints['left_thumb_4'] is None:
            hand_keypoints.pop('left_thumb_4')
        elif body_keypoints['left_thumb_4'] is None and hand_keypoints['left_thumb_4'] is not None:
            body_keypoints.pop('left_thumb_4')
        else:
            hand_keypoints.pop('left_thumb_4')
        full_body_keypoints = {**body_keypoints, **hand_keypoints}
        return full_body_keypoints

class Open_Pose_Infer_Full(object):
    def __init__(self) -> None:
        params = dict()
        params["model_folder"] = r"backends\open_pose\build\models"
        params["hand"] = True
        params["hand_detector"] = 0
        params["body"] = 1
        params["net_resolution_dynamic"] = 1
        params["net_resolution"] = "-656x256" #"-1x256"
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        self.body_keypoint_names = ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow',
                                    'left_wrist', 'MidHip', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
                                    'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel',
                                    'right_big_toe', 'right_small_toe', 'right_heel']
        self.right_hand_keypoint_names = ['right_palm', 'right_thumb_4', 'right_thumb_3', 'right_thumb_2', 'right_thumb', 'right_index_finger_4', 'right_index_finger_3',
                                          'right_index_finger_2', 'right_index_finger', 'right_middle_finger_4', 'right_middle_finger_3', 'right_middle_finger_2',
                                          'right_middle_finger', 'right_ring_finger_4', 'right_ring_finger_3', 'right_ring_finger_2', 'right_ring_finger',
                                          'right_baby_finger_4', 'right_baby_finger_3', 'right_baby_finger_2', 'right_baby_finger']
        self.left_hand_keypoint_names = ['left_palm', 'left_thumb_4', 'left_thumb_3', 'left_thumb_2', 'left_thumb', 'left_index_finger_4', 'left_index_finger_3',
                                          'left_index_finger_2', 'left_index_finger', 'left_middle_finger_4', 'left_middle_finger_3', 'left_middle_finger_2',
                                          'left_middle_finger', 'left_ring_finger_4', 'left_ring_finger_3', 'left_ring_finger_2', 'left_ring_finger',
                                          'left_baby_finger_4', 'left_baby_finger_3', 'left_baby_finger_2', 'left_baby_finger']

    def execute(self, image):
        datum = op.Datum()
        datum.cvInputData = image
        try:
            self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            body_keypoints_pixel_locations = datum.poseKeypoints[0]
            right_hand_keypoints_pixel_locations = datum.handKeypoints[1][0]
            left_hand_keypoints_pixel_locations = datum.handKeypoints[0][0]
            body_keypoints_cordinates = [(element[0], element[1]) if (element[0]!=0.0 and element[1]!=0.0) else None for element in body_keypoints_pixel_locations]
            right_hand_keypoints_cordinates = [(element[0], element[1]) if (element[0]!=0.0 and element[1]!=0.0) else None for element in right_hand_keypoints_pixel_locations]
            left_hand_keypoints_cordinates = [(element[0], element[1]) if (element[0]!=0.0 and element[1]!=0.0) else None for element in left_hand_keypoints_pixel_locations]
            body_keypoints = dict(zip(self.body_keypoint_names, body_keypoints_cordinates))
            right_hand_keypoints = dict(zip(self.right_hand_keypoint_names, right_hand_keypoints_cordinates))
            left_hand_keypoints = dict(zip(self.left_hand_keypoint_names, left_hand_keypoints_cordinates))
        except:
            print("OpenPose Error: No person in the frame!")
            body_keypoints = dict.fromkeys(self.body_keypoint_names)
            right_hand_keypoints = dict.fromkeys(self.right_hand_keypoint_names)
            left_hand_keypoints = dict.fromkeys(self.left_hand_keypoint_names)
        all_keypoints = {**body_keypoints, **right_hand_keypoints, **left_hand_keypoints}
        return all_keypoints


class TRT_Infer_Body(object):
    def __init__(self) -> None:
        body_pose_path = os.path.join("backends", "trt_pose", "body_pose", "human_pose.json")
        body_model_weights = os.path.join("backends", "trt_pose", "body_pose", "resnet18_baseline_att_224x224_A_epoch_249.pth")
        with open(body_pose_path, 'r') as f:
            body_pose = json.load(f)
        self.body_topology = trt_pose.coco.coco_category_to_topology(body_pose)
        num_body_parts = len(body_pose['keypoints'])
        num_body_links = len(body_pose['skeleton'])
        self.body_keypoint_names = body_pose['keypoints']
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.body_model = trt_pose.models.resnet18_baseline_att(num_body_parts, 2 * num_body_links).cuda().eval()
            self.body_model.load_state_dict(torch.load(body_model_weights))
            self.body_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
            self.body_std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        else:
            self.device = torch.device('cpu')
            self.body_model = trt_pose.models.resnet18_baseline_att(num_body_parts, 2 * num_body_links).cpu().eval()
            self.body_model.load_state_dict(torch.load(body_model_weights, map_location=torch.device('cpu')))
            self.body_mean = torch.Tensor([0.485, 0.456, 0.406]).cpu()
            self.body_std = torch.Tensor([0.229, 0.224, 0.225]).cpu()
        self.parse_body_objects = ParseObjects(self.body_topology)
        self.body_model_target_shape = (224, 224)

    def execute(self, img):
        data = self.preprocess(img)
        cmap, paf = self.body_model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        _, objects, peaks = self.parse_body_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        object = objects[0][0]
        body_keypoint_coordinates = self.get_positions(img, object, peaks)
        return body_keypoint_coordinates

    def execute_with_queue(self, que, subject, img):
        data = self.preprocess(img)
        cmap, paf = self.body_model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        _, objects, peaks = self.parse_body_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        object = objects[0][0]
        body_keypoint_coordinates = self.get_positions(img, object, peaks)
        que.put({subject+'_body': body_keypoint_coordinates})

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image).resize(self.body_model_target_shape)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.body_mean[:, None, None]).div_(self.body_std[:, None, None])
        return image[None, ...]

    def get_positions(self, image, obj, normalized_peaks):
        height = image.shape[0]
        width = image.shape[1]
        positions = {}
        C = obj.shape[0]
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                peak = normalized_peaks[0][j][k]
                x = round(float(peak[1]) * width)
                y = round(float(peak[0]) * height)
                positions[self.body_keypoint_names[j]] = (x, y)
            else:
                positions[self.body_keypoint_names[j]] = None
        return positions

    def parse_keypoints(self, obj, keypoints, part_to_draw):
        joints = self.body_topology.shape[0]
        counts = obj.shape[0]
        if part_to_draw == 'full':
            positions = list(range(counts))
            joints_to_draw = list(range(joints))
        elif part_to_draw == 'upper':
            positions = list(range(counts-5))
            positions.append(counts-1)
            joints_to_draw = list(range(4, joints))
        elif part_to_draw == 'lower':
            positions = list(range(counts-7, counts-1))
            joints_to_draw = list(range(5))
        coordinates = list(keypoints.values())
        selected_coordinates = [coordinates[i] for i in positions]
        return selected_coordinates, joints_to_draw


class TRT_Infer_Hands(object):
    def __init__(self) -> None:
        hand_pose_path = os.path.join("backends", "trt_pose", "hand_pose", "preprocess", "hand_pose.json")
        hand_model_weights =  os.path.join("backends", "trt_pose", "hand_pose", "trained_models", "hand_pose_resnet18_att_244_244.pth")
        with open(hand_pose_path, 'r') as f:
            hand_pose = json.load(f)
        self.hand_topology = trt_pose.coco.coco_category_to_topology(hand_pose)
        num_hand_parts = len(hand_pose['keypoints'])
        num_hand_links = len(hand_pose['skeleton'])
        self.hand_keypoint_names = hand_pose['keypoints']
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.hand_model = trt_pose.models.resnet18_baseline_att(num_hand_parts, 2 * num_hand_links).cuda().eval()
            self.hand_model.load_state_dict(torch.load(hand_model_weights))
            self.hand_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
            self.hand_std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        else:
            self.device = torch.device('cpu')
            self.hand_model = trt_pose.models.resnet18_baseline_att(num_hand_parts, 2 * num_hand_links).cpu().eval()
            self.hand_model.load_state_dict(torch.load(hand_model_weights, map_location=torch.device('cpu')))
            self.hand_mean = torch.Tensor([0.485, 0.456, 0.406]).cpu()
            self.hand_std = torch.Tensor([0.229, 0.224, 0.225]).cpu()
        self.parse_hand_objects = ParseObjects(self.hand_topology, cmap_threshold=0.15, link_threshold=0.15)
        self.hand_model_target_shape = (224, 224)

    def execute(self, image):
        data = self.preprocess(image)
        cmap, paf = self.hand_model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        count, objects, peaks = self.parse_hand_objects(cmap, paf)
        hands_keypoint_coordinates = self.get_positions(image, count, objects, peaks)
        return hands_keypoint_coordinates

    def execute_with_queue(self, que, subject, image):
        data = self.preprocess(image)
        cmap, paf = self.hand_model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        count, objects, peaks = self.parse_hand_objects(cmap, paf)
        hands_keypoint_coordinates = self.get_positions(image, count, objects, peaks)
        que.put({subject+'_hands': hands_keypoint_coordinates})

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image).resize(self.hand_model_target_shape)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.hand_mean[:, None, None]).div_(self.hand_std[:, None, None])
        return image[None, ...]

    def get_positions(self, image, count, objects, normalized_peaks):
        height = image.shape[0]
        width = image.shape[1]
        both_hands_positions = {}
        count = count[0] if count[0]<3 else 2
        if count==0:
            both_hands_positions["hand_0"] = None
            both_hands_positions["hand_1"] = None
        elif count==1:
            obj = objects[0][0]
            C = obj.shape[0]
            per_hand_positions = {}
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    per_hand_positions[self.hand_keypoint_names[j]] = (x, y)
                else:
                    per_hand_positions[self.hand_keypoint_names[j]] = None
            both_hands_positions["hand_0"] = per_hand_positions
            both_hands_positions["hand_1"] = None
        else:
            for i in range(count):
                obj = objects[0][i]
                C = obj.shape[0]
                per_hand_positions = {}
                for j in range(C):
                    k = int(obj[j])
                    if k >= 0:
                        peak = normalized_peaks[0][j][k]
                        x = round(float(peak[1]) * width)
                        y = round(float(peak[0]) * height)
                        per_hand_positions[self.hand_keypoint_names[j]] = (x, y)
                    else:
                        per_hand_positions[self.hand_keypoint_names[j]] = None
                both_hands_positions[f"hand_{i}"] = per_hand_positions
        return both_hands_positions


class TRT_Infer_Feet(object):
    def __init__(self) -> None:
        self.upsample_ratio = 4
        self.net_input_height_size = 256
        self.pad_value=(0, 0, 0)
        self.img_mean=(128, 128, 128)
        self.img_scale=1/256
        self.stride = 8
        self.feet_keypoint_names = ['r_ank', 'l_ank', 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel']
        foot_model_weights = os.path.join("backends", "trt_pose", "foot_pose", "trained_models", "checkpoint_iter_384000.pth")
        self.foot_model = PoseEstimationWithMobileNet(num_heatmaps=25, num_pafs=50)
        if torch.cuda.is_available():
            self.cuda = True
            checkpoint = torch.load(foot_model_weights)
            load_state(self.foot_model, checkpoint)
            self.foot_model.cuda().eval()
        else:
            self.cuda = False
            checkpoint = torch.load(foot_model_weights, map_location=torch.device('cpu'))
            load_state(self.foot_model, checkpoint)
            self.foot_model.cpu().eval()

    def execute(self, image):
        tensor_img, pad, scale = self.preprocess(image=image)
        stages_output = self.foot_model(tensor_img)
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)
        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)
        current_poses = self.postprocess(heatmaps=heatmaps, pafs=pafs, pad=pad, scale=scale)
        feet_keypoint_coordinates = self.parse_poses(current_poses)
        return feet_keypoint_coordinates

    def execute_with_queue(self, que, subject, image):
        tensor_img, pad, scale = self.preprocess(image=image)
        stages_output = self.foot_model(tensor_img)
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)
        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)
        current_poses = self.postprocess(heatmaps=heatmaps, pafs=pafs, pad=pad, scale=scale)
        feet_keypoint_coordinates = self.parse_poses(current_poses)
        que.put({subject+'_feet': feet_keypoint_coordinates})

    def preprocess(self, image):
        height, _, _ = image.shape
        scale = self.net_input_height_size / height
        scaled_img = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = normalize(scaled_img, self.img_mean, self.img_scale)
        min_dims = [self.net_input_height_size, max(scaled_img.shape[1], self.net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, self.stride, self.pad_value, min_dims)
        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if self.cuda:
            tensor_img = tensor_img.cuda()
        else:
            tensor_img = tensor_img.cpu()
        return tensor_img, pad, scale

    def postprocess(self, heatmaps, pafs, pad, scale):
        total_keypoints_num = 0
        all_keypoints_by_type = []
        num_keypoints = Pose.num_kpts
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
        return current_poses

    def parse_poses(self, poses):
        positions = dict.fromkeys(self.feet_keypoint_names)
        if len(poses)<=2:
            for pose in poses:
                foot_keypoints = pose.parse_feet_keypoints()
                for index, foot_keypoint in enumerate(foot_keypoints):
                    if foot_keypoint:
                        positions[self.feet_keypoint_names[index]] = foot_keypoint
        else:
            right_foot = None
            left_foot = None
            for pose in poses:
                foot_keypoints = pose.parse_feet_keypoints()
                if foot_keypoints[0] is None and foot_keypoints[5] is None and foot_keypoints[6] is None and foot_keypoints[6] is None:
                    if left_foot is None:
                        left_foot = foot_keypoints
                else:
                    if right_foot is None:
                        right_foot = foot_keypoints
                if right_foot is not None and left_foot is not None:
                    break
            if right_foot is not None:
                for index, foot_keypoint in enumerate(right_foot):
                        if foot_keypoint:
                            positions[self.feet_keypoint_names[index]] = foot_keypoint
            if left_foot is not None:
                for index, foot_keypoint in enumerate(left_foot):
                        if foot_keypoint:
                            positions[self.feet_keypoint_names[index]] = foot_keypoint
        return positions


right_keys_to_check_for_hand = ['right_wrist', 'right_elbow', 'right_shoulder']
left_keys_to_check_for_hand = ['left_wrist', 'left_elbow', 'left_shoulder']
hand_keypoint_names = ["palm","thumb", "thumb_2", "thumb_3", "thumb_4", "index_finger", "index_finger_2", "index_finger_3",
                        "index_finger_4", "middle_finger", "middle_finger_2", "middle_finger_3", "middle_finger_4",
                        "ring_finger", "ring_finger_2", "ring_finger_3", "ring_finger_4", "baby_finger", "baby_finger_2",
                        "baby_finger_3", "baby_finger_4"]

def join_trt_keypoints(body_keypoints, hands_keypoints=None, feet_keypoints=None):
    if hands_keypoints is not None:
        right_body_hand_anchor = None
        left_body_hand_anchor = None
        hands = None
        for key_to_check in right_keys_to_check_for_hand:
            if body_keypoints[key_to_check] is not None:
                right_body_hand_anchor = key_to_check
                break
        for key_to_check in left_keys_to_check_for_hand:
            if body_keypoints[key_to_check] is not None:
                left_body_hand_anchor = key_to_check
                break

        if hands_keypoints['hand_0'] is not None and right_body_hand_anchor is not None and left_body_hand_anchor is not None:
            first_hand = hands_keypoints['hand_0']
            for key in first_hand:
                if first_hand[key] is not None:
                    right_side_distance = euclidean(first_hand[key], body_keypoints[right_body_hand_anchor])
                    left_side_distance = euclidean(first_hand[key], body_keypoints[left_body_hand_anchor])
                    if right_side_distance<left_side_distance: # Hand is right
                        right_hand = dict(("right_"+ key, value) for (key, value) in first_hand.items())
                        if hands_keypoints['hand_1'] is not None:
                            left_hand = dict(("left_"+ key, value) for (key, value) in hands_keypoints['hand_1'].items())
                        else:
                            left_hand = dict(("left_"+ key, None) for (key, _) in first_hand.items())
                    else:
                        left_hand = dict(("left_"+ key, value) for (key, value) in first_hand.items())
                        if hands_keypoints['hand_1'] is not None:
                            right_hand = dict(("right_"+ key, value) for (key, value) in hands_keypoints['hand_1'].items())
                        else:
                            right_hand = dict(("right_"+ key, None) for (key, _) in first_hand.items())
                    hands = {**right_hand, **left_hand}
                    break
        elif hands_keypoints['hand_0'] is not None and hands_keypoints['hand_1'] is not None and right_body_hand_anchor is not None:
            first_hand = hands_keypoints['hand_0']
            second_hand = hands_keypoints['hand_1']
            hands = None
            for key in first_hand:
                if first_hand[key] is not None and second_hand[key] is not None:
                    one_side_distance = euclidean(first_hand[key], body_keypoints[right_body_hand_anchor])
                    second_side_distance = euclidean(second_hand[key], body_keypoints[right_body_hand_anchor])
                    if one_side_distance<second_side_distance:
                        right_hand = dict(("right_"+ key, value) for (key, value) in first_hand.items())
                        left_hand = dict(("left_"+ key, value) for (key, value) in second_hand.items())
                    else:
                        right_hand = dict(("right_"+ key, value) for (key, value) in second_hand.items())
                        left_hand = dict(("left_"+ key, value) for (key, value) in first_hand.items())
                    hands = {**right_hand, **left_hand}
                    break
            if hands is None:
                right_hand = dict(("right_"+ key, None) for (key, _) in hands_keypoints['hand_0'].items())
                left_hand = dict(("left_"+ key, None) for (key, _) in hands_keypoints['hand_0'].items())
                hands = {**right_hand, **left_hand}
        elif hands_keypoints['hand_0'] is not None and hands_keypoints['hand_1'] is not None and left_body_hand_anchor is not None:
            first_hand = hands_keypoints['hand_0']
            second_hand = hands_keypoints['hand_1']
            hands = None
            for key in first_hand:
                if first_hand[key] is not None and second_hand[key] is not None:
                    one_side_distance = euclidean(first_hand[key], body_keypoints[left_body_hand_anchor])
                    second_side_distance = euclidean(second_hand[key], body_keypoints[left_body_hand_anchor])
                    if one_side_distance<second_side_distance:
                        right_hand = dict(("right_"+ key, value) for (key, value) in second_hand.items())
                        left_hand = dict(("left_"+ key, value) for (key, value) in first_hand.items())
                    else:
                        right_hand = dict(("right_"+ key, value) for (key, value) in first_hand.items())
                        left_hand = dict(("left_"+ key, value) for (key, value) in second_hand.items())
                    hands = {**right_hand, **left_hand}
                    break
            if hands is None:
                right_hand = dict(("right_"+ key, None) for (key, _) in hands_keypoints['hand_0'].items())
                left_hand = dict(("left_"+ key, None) for (key, _) in hands_keypoints['hand_0'].items())
                hands = {**right_hand, **left_hand}
        else:
            hand_keypoints = dict.fromkeys(hand_keypoint_names)
            right_hand = dict(("right_"+ key, None) for (key, _) in hand_keypoints.items())
            left_hand = dict(("left_"+ key, None) for (key, _) in hand_keypoints.items())
            hands = {**right_hand, **left_hand}
    if feet_keypoints is not None:
        if feet_keypoints['r_ank'] is not None and feet_keypoints['l_ank'] is not None:
            if body_keypoints['right_ankle'] is not None:
                bodyRightAnkle_to_right_feet_distance = euclidean(body_keypoints['right_ankle'], feet_keypoints['r_ank'])
                bodyRightAnkle_to_left_feet_distance = euclidean(body_keypoints['right_ankle'], feet_keypoints['l_ank'])
                if bodyRightAnkle_to_right_feet_distance<bodyRightAnkle_to_left_feet_distance:
                    pass
                else:
                    feet_keypoints = swap_feet(feet_keypoints)
            elif body_keypoints['left_ankle'] is not None:
                bodyleftAnkle_to_right_feet_distance = euclidean(body_keypoints['left_ankle'], feet_keypoints['r_ank'])
                bodyleftAnkle_to_left_feet_distance = euclidean(body_keypoints['left_ankle'], feet_keypoints['l_ank'])
                if bodyleftAnkle_to_left_feet_distance<bodyleftAnkle_to_right_feet_distance:
                    pass
                else:
                    feet_keypoints = swap_feet(feet_keypoints)

            elif body_keypoints['right_knee'] is not None:
                bodyRightknee_to_right_feet_distance = euclidean(body_keypoints['right_knee'], feet_keypoints['r_ank'])
                bodyRightknee_to_left_feet_distance = euclidean(body_keypoints['right_knee'], feet_keypoints['l_ank'])
                if bodyRightknee_to_right_feet_distance<bodyRightknee_to_left_feet_distance:
                    pass
                else:
                    feet_keypoints = swap_feet(feet_keypoints)
            elif body_keypoints['left_knee'] is not None:
                bodyleftknee_to_right_feet_distance = euclidean(body_keypoints['left_knee'], feet_keypoints['r_ank'])
                bodyleftknee_to_left_feet_distance = euclidean(body_keypoints['left_knee'], feet_keypoints['l_ank'])
                if bodyleftknee_to_left_feet_distance<bodyleftknee_to_right_feet_distance:
                    pass
                else:
                    feet_keypoints = swap_feet(feet_keypoints)
            else:
                feet_keypoints = dict.fromkeys([*feet_keypoints])
        else:
            right_mean, left_mean = get_means(feet_keypoints)
            if right_mean is None or left_mean is None:
                feet_keypoints = dict.fromkeys([*feet_keypoints])
            else:
                if body_keypoints['right_ankle'] is not None:
                    bodyRightAnkle_to_right_feet_distance = euclidean(body_keypoints['right_ankle'], right_mean)
                    bodyRightAnkle_to_left_feet_distance = euclidean(body_keypoints['right_ankle'], left_mean)
                    if bodyRightAnkle_to_right_feet_distance<bodyRightAnkle_to_left_feet_distance:
                        pass
                    else:
                        feet_keypoints = swap_feet(feet_keypoints)
                elif body_keypoints['left_ankle'] is not None:
                    bodyleftAnkle_to_right_feet_distance = euclidean(body_keypoints['left_ankle'], right_mean)
                    bodyleftAnkle_to_left_feet_distance = euclidean(body_keypoints['left_ankle'], left_mean)
                    if bodyleftAnkle_to_left_feet_distance<bodyleftAnkle_to_right_feet_distance:
                        pass
                    else:
                        feet_keypoints = swap_feet(feet_keypoints)
                elif body_keypoints['right_knee'] is not None:
                    bodyRightknee_to_right_feet_distance = euclidean(body_keypoints['right_knee'], right_mean)
                    bodyRightknee_to_left_feet_distance = euclidean(body_keypoints['right_knee'], left_mean)
                    if bodyRightknee_to_right_feet_distance<bodyRightknee_to_left_feet_distance:
                        pass
                    else:
                        feet_keypoints = swap_feet(feet_keypoints)
                elif body_keypoints['left_knee'] is not None:
                    bodyleftknee_to_right_feet_distance = euclidean(body_keypoints['left_knee'], right_mean)
                    bodyleftknee_to_left_feet_distance = euclidean(body_keypoints['left_knee'], left_mean)
                    if bodyleftknee_to_left_feet_distance<bodyleftknee_to_right_feet_distance:
                        pass
                    else:
                        feet_keypoints = swap_feet(feet_keypoints)
                else:
                    feet_keypoints = dict.fromkeys([*feet_keypoints])

    if hands_keypoints is not None and feet_keypoints is not None:
        joined_keypoints = {**body_keypoints, **hands, **feet_keypoints}
    elif hands_keypoints is not None:
        joined_keypoints = {**body_keypoints, **hands}
    elif feet_keypoints is not None:
        joined_keypoints = {**body_keypoints, **feet_keypoints}
    else:
        joined_keypoints = body_keypoints
    if hands_keypoints is not None:
        if joined_keypoints['right_wrist'] is not None and joined_keypoints['right_palm'] is not None:
            joined_keypoints.pop('right_palm')
        elif joined_keypoints['right_wrist'] is None and joined_keypoints['right_palm'] is not None:
            joined_keypoints.pop('right_wrist')
            joined_keypoints['right_wrist'] = joined_keypoints.pop('right_palm')
        elif joined_keypoints['right_wrist'] is not None and joined_keypoints['right_palm'] is None:
            joined_keypoints.pop('right_palm')
        else:
            joined_keypoints.pop('right_palm')
        if joined_keypoints['left_wrist'] is not None and joined_keypoints['left_palm'] is not None:
            joined_keypoints.pop('left_palm')
        elif joined_keypoints['left_wrist'] is None and joined_keypoints['left_palm'] is not None:
            joined_keypoints.pop('left_wrist')
            joined_keypoints['left_wrist'] = joined_keypoints.pop('left_palm')
        elif joined_keypoints['left_wrist'] is not None and joined_keypoints['left_palm'] is None:
            joined_keypoints.pop('left_palm')
        else:
            joined_keypoints.pop('left_palm')
    if feet_keypoints is not None:
        if joined_keypoints['right_ankle'] is not None and joined_keypoints['r_ank'] is not None:
            joined_keypoints.pop('r_ank')
        elif joined_keypoints['right_ankle'] is None and joined_keypoints['r_ank'] is not None:
            joined_keypoints.pop('right_ankle')
            joined_keypoints['right_ankle'] = joined_keypoints.pop('r_ank')
        elif joined_keypoints['right_ankle'] is not None and joined_keypoints['r_ank'] is None:
            joined_keypoints.pop('r_ank')
        else:
            joined_keypoints.pop('r_ank')
        if joined_keypoints['left_ankle'] is not None and joined_keypoints['l_ank'] is not None:
            joined_keypoints.pop('l_ank')
        elif joined_keypoints['left_ankle'] is None and joined_keypoints['l_ank'] is not None:
            joined_keypoints.pop('left_ankle')
            joined_keypoints['left_ankle'] = joined_keypoints.pop('l_ank')
        elif joined_keypoints['left_ankle'] is not None and joined_keypoints['l_ank'] is None:
            joined_keypoints.pop('l_ank')
        else:
            joined_keypoints.pop('l_ank')
    return joined_keypoints


def join_trt_keypoints_with_queue(que, subject, body_keypoints, hands_keypoints=None, feet_keypoints=None):
    if hands_keypoints is not None:
        right_body_hand_anchor = None
        left_body_hand_anchor = None
        hands = None
        for key_to_check in right_keys_to_check_for_hand:
            if body_keypoints[key_to_check] is not None:
                right_body_hand_anchor = key_to_check
                break
        for key_to_check in left_keys_to_check_for_hand:
            if body_keypoints[key_to_check] is not None:
                left_body_hand_anchor = key_to_check
                break

        if hands_keypoints['hand_0'] is not None and right_body_hand_anchor is not None and left_body_hand_anchor is not None:
            first_hand = hands_keypoints['hand_0']
            for key in first_hand:
                if first_hand[key] is not None:
                    right_side_distance = euclidean(first_hand[key], body_keypoints[right_body_hand_anchor])
                    left_side_distance = euclidean(first_hand[key], body_keypoints[left_body_hand_anchor])
                    if right_side_distance<left_side_distance: # Hand is right
                        right_hand = dict(("right_"+ key, value) for (key, value) in first_hand.items())
                        if hands_keypoints['hand_1'] is not None:
                            left_hand = dict(("left_"+ key, value) for (key, value) in hands_keypoints['hand_1'].items())
                        else:
                            left_hand = dict(("left_"+ key, None) for (key, _) in first_hand.items())
                    else:
                        left_hand = dict(("left_"+ key, value) for (key, value) in first_hand.items())
                        if hands_keypoints['hand_1'] is not None:
                            right_hand = dict(("right_"+ key, value) for (key, value) in hands_keypoints['hand_1'].items())
                        else:
                            right_hand = dict(("right_"+ key, None) for (key, _) in first_hand.items())
                    hands = {**right_hand, **left_hand}
                    break
        elif hands_keypoints['hand_0'] is not None and hands_keypoints['hand_1'] is not None and right_body_hand_anchor is not None:
            first_hand = hands_keypoints['hand_0']
            second_hand = hands_keypoints['hand_1']
            hands = None
            for key in first_hand:
                if first_hand[key] is not None and second_hand[key] is not None:
                    one_side_distance = euclidean(first_hand[key], body_keypoints[right_body_hand_anchor])
                    second_side_distance = euclidean(second_hand[key], body_keypoints[right_body_hand_anchor])
                    if one_side_distance<second_side_distance:
                        right_hand = dict(("right_"+ key, value) for (key, value) in first_hand.items())
                        left_hand = dict(("left_"+ key, value) for (key, value) in second_hand.items())
                    else:
                        right_hand = dict(("right_"+ key, value) for (key, value) in second_hand.items())
                        left_hand = dict(("left_"+ key, value) for (key, value) in first_hand.items())
                    hands = {**right_hand, **left_hand}
                    break
            if hands is None:
                right_hand = dict(("right_"+ key, None) for (key, _) in hands_keypoints['hand_0'].items())
                left_hand = dict(("left_"+ key, None) for (key, _) in hands_keypoints['hand_0'].items())
                hands = {**right_hand, **left_hand}
        elif hands_keypoints['hand_0'] is not None and hands_keypoints['hand_1'] is not None and left_body_hand_anchor is not None:
            first_hand = hands_keypoints['hand_0']
            second_hand = hands_keypoints['hand_1']
            hands = None
            for key in first_hand:
                if first_hand[key] is not None and second_hand[key] is not None:
                    one_side_distance = euclidean(first_hand[key], body_keypoints[left_body_hand_anchor])
                    second_side_distance = euclidean(second_hand[key], body_keypoints[left_body_hand_anchor])
                    if one_side_distance<second_side_distance:
                        right_hand = dict(("right_"+ key, value) for (key, value) in second_hand.items())
                        left_hand = dict(("left_"+ key, value) for (key, value) in first_hand.items())
                    else:
                        right_hand = dict(("right_"+ key, value) for (key, value) in first_hand.items())
                        left_hand = dict(("left_"+ key, value) for (key, value) in second_hand.items())
                    hands = {**right_hand, **left_hand}
                    break
            if hands is None:
                right_hand = dict(("right_"+ key, None) for (key, _) in hands_keypoints['hand_0'].items())
                left_hand = dict(("left_"+ key, None) for (key, _) in hands_keypoints['hand_0'].items())
                hands = {**right_hand, **left_hand}
        else:
            hand_keypoints = dict.fromkeys(hand_keypoint_names)
            right_hand = dict(("right_"+ key, None) for (key, _) in hand_keypoints.items())
            left_hand = dict(("left_"+ key, None) for (key, _) in hand_keypoints.items())
            hands = {**right_hand, **left_hand}
    if feet_keypoints is not None:
        if feet_keypoints['r_ank'] is not None and feet_keypoints['l_ank'] is not None:
            if body_keypoints['right_ankle'] is not None:
                bodyRightAnkle_to_right_feet_distance = euclidean(body_keypoints['right_ankle'], feet_keypoints['r_ank'])
                bodyRightAnkle_to_left_feet_distance = euclidean(body_keypoints['right_ankle'], feet_keypoints['l_ank'])
                if bodyRightAnkle_to_right_feet_distance<bodyRightAnkle_to_left_feet_distance:
                    pass
                else:
                    feet_keypoints = swap_feet(feet_keypoints)
            elif body_keypoints['left_ankle'] is not None:
                bodyleftAnkle_to_right_feet_distance = euclidean(body_keypoints['left_ankle'], feet_keypoints['r_ank'])
                bodyleftAnkle_to_left_feet_distance = euclidean(body_keypoints['left_ankle'], feet_keypoints['l_ank'])
                if bodyleftAnkle_to_left_feet_distance<bodyleftAnkle_to_right_feet_distance:
                    pass
                else:
                    feet_keypoints = swap_feet(feet_keypoints)

            elif body_keypoints['right_knee'] is not None:
                bodyRightknee_to_right_feet_distance = euclidean(body_keypoints['right_knee'], feet_keypoints['r_ank'])
                bodyRightknee_to_left_feet_distance = euclidean(body_keypoints['right_knee'], feet_keypoints['l_ank'])
                if bodyRightknee_to_right_feet_distance<bodyRightknee_to_left_feet_distance:
                    pass
                else:
                    feet_keypoints = swap_feet(feet_keypoints)
            elif body_keypoints['left_knee'] is not None:
                bodyleftknee_to_right_feet_distance = euclidean(body_keypoints['left_knee'], feet_keypoints['r_ank'])
                bodyleftknee_to_left_feet_distance = euclidean(body_keypoints['left_knee'], feet_keypoints['l_ank'])
                if bodyleftknee_to_left_feet_distance<bodyleftknee_to_right_feet_distance:
                    pass
                else:
                    feet_keypoints = swap_feet(feet_keypoints)
            else:
                feet_keypoints = dict.fromkeys([*feet_keypoints])
        else:
            right_mean, left_mean = get_means(feet_keypoints)
            if right_mean is None or left_mean is None:
                feet_keypoints = dict.fromkeys([*feet_keypoints])
            else:
                if body_keypoints['right_ankle'] is not None:
                    bodyRightAnkle_to_right_feet_distance = euclidean(body_keypoints['right_ankle'], right_mean)
                    bodyRightAnkle_to_left_feet_distance = euclidean(body_keypoints['right_ankle'], left_mean)
                    if bodyRightAnkle_to_right_feet_distance<bodyRightAnkle_to_left_feet_distance:
                        pass
                    else:
                        feet_keypoints = swap_feet(feet_keypoints)
                elif body_keypoints['left_ankle'] is not None:
                    bodyleftAnkle_to_right_feet_distance = euclidean(body_keypoints['left_ankle'], right_mean)
                    bodyleftAnkle_to_left_feet_distance = euclidean(body_keypoints['left_ankle'], left_mean)
                    if bodyleftAnkle_to_left_feet_distance<bodyleftAnkle_to_right_feet_distance:
                        pass
                    else:
                        feet_keypoints = swap_feet(feet_keypoints)
                elif body_keypoints['right_knee'] is not None:
                    bodyRightknee_to_right_feet_distance = euclidean(body_keypoints['right_knee'], right_mean)
                    bodyRightknee_to_left_feet_distance = euclidean(body_keypoints['right_knee'], left_mean)
                    if bodyRightknee_to_right_feet_distance<bodyRightknee_to_left_feet_distance:
                        pass
                    else:
                        feet_keypoints = swap_feet(feet_keypoints)
                elif body_keypoints['left_knee'] is not None:
                    bodyleftknee_to_right_feet_distance = euclidean(body_keypoints['left_knee'], right_mean)
                    bodyleftknee_to_left_feet_distance = euclidean(body_keypoints['left_knee'], left_mean)
                    if bodyleftknee_to_left_feet_distance<bodyleftknee_to_right_feet_distance:
                        pass
                    else:
                        feet_keypoints = swap_feet(feet_keypoints)
                else:
                    feet_keypoints = dict.fromkeys([*feet_keypoints])

    if hands_keypoints is not None and feet_keypoints is not None:
        joined_keypoints = {**body_keypoints, **hands, **feet_keypoints}
    elif hands_keypoints is not None:
        joined_keypoints = {**body_keypoints, **hands}
    elif feet_keypoints is not None:
        joined_keypoints = {**body_keypoints, **feet_keypoints}
    else:
        joined_keypoints = body_keypoints
    if hands_keypoints is not None:
        if joined_keypoints['right_wrist'] is not None and joined_keypoints['right_palm'] is not None:
            joined_keypoints.pop('right_palm')
        elif joined_keypoints['right_wrist'] is None and joined_keypoints['right_palm'] is not None:
            joined_keypoints.pop('right_wrist')
            joined_keypoints['right_wrist'] = joined_keypoints.pop('right_palm')
        elif joined_keypoints['right_wrist'] is not None and joined_keypoints['right_palm'] is None:
            joined_keypoints.pop('right_palm')
        else:
            joined_keypoints.pop('right_palm')
        if joined_keypoints['left_wrist'] is not None and joined_keypoints['left_palm'] is not None:
            joined_keypoints.pop('left_palm')
        elif joined_keypoints['left_wrist'] is None and joined_keypoints['left_palm'] is not None:
            joined_keypoints.pop('left_wrist')
            joined_keypoints['left_wrist'] = joined_keypoints.pop('left_palm')
        elif joined_keypoints['left_wrist'] is not None and joined_keypoints['left_palm'] is None:
            joined_keypoints.pop('left_palm')
        else:
            joined_keypoints.pop('left_palm')
    if feet_keypoints is not None:
        if joined_keypoints['right_ankle'] is not None and joined_keypoints['r_ank'] is not None:
            joined_keypoints.pop('r_ank')
        elif joined_keypoints['right_ankle'] is None and joined_keypoints['r_ank'] is not None:
            joined_keypoints.pop('right_ankle')
            joined_keypoints['right_ankle'] = joined_keypoints.pop('r_ank')
        elif joined_keypoints['right_ankle'] is not None and joined_keypoints['r_ank'] is None:
            joined_keypoints.pop('r_ank')
        else:
            joined_keypoints.pop('r_ank')
        if joined_keypoints['left_ankle'] is not None and joined_keypoints['l_ank'] is not None:
            joined_keypoints.pop('l_ank')
        elif joined_keypoints['left_ankle'] is None and joined_keypoints['l_ank'] is not None:
            joined_keypoints.pop('left_ankle')
            joined_keypoints['left_ankle'] = joined_keypoints.pop('l_ank')
        elif joined_keypoints['left_ankle'] is not None and joined_keypoints['l_ank'] is None:
            joined_keypoints.pop('l_ank')
        else:
            joined_keypoints.pop('l_ank')
    que.put({subject: joined_keypoints})

def get_means(feet_keypoints):
    feet_coordinates = list(feet_keypoints.values())
    right_foot = [feet_coordinates[i] for i in [0, 5, 6, 7]]
    left_foot = feet_coordinates[1:5]
    if right_foot.count(None) != len(right_foot):
        rightX = mean([item[0] for item in right_foot if item is not None])
        rightY = mean([item[1] for item in right_foot if item is not None])
        right_mean = (rightX, rightY)
    else:
        right_mean = None
    if left_foot.count(None) != len(left_foot):
        leftX = mean([item[0] for item in left_foot if item is not None])
        leftY = mean([item[1] for item in left_foot if item is not None])
        left_mean = (leftX, leftY)
    else:
        left_mean = None
    return right_mean, left_mean

def swap_feet(feet_keypoints):
    swaped_keypoints = dict.fromkeys([*feet_keypoints])
    for key in swaped_keypoints:
        if 'r_' in key:
            swaping_key = key.replace('r_a', 'l_a')
        elif 'l_' in key:
            swaping_key = key.replace('l_a', 'r_a')
        elif 'left_' in key:
            swaping_key = key.replace('left_', 'right_')
        elif 'right_' in key:
            swaping_key = key.replace('right_', 'left_')
        swaped_keypoints[key] = feet_keypoints[swaping_key]
    return swaped_keypoints

ordered_keypoint_names = ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow',
                        'left_wrist', 'MidHip', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
                        'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe',
                        'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'right_palm', 'right_thumb_4',
                        'right_thumb_3', 'right_thumb_2', 'right_thumb', 'right_index_finger_4', 'right_index_finger_3',
                        'right_index_finger_2', 'right_index_finger', 'right_middle_finger_4', 'right_middle_finger_3',
                        'right_middle_finger_2', 'right_middle_finger', 'right_ring_finger_4', 'right_ring_finger_3',
                        'right_ring_finger_2', 'right_ring_finger', 'right_baby_finger_4', 'right_baby_finger_3',
                        'right_baby_finger_2', 'right_baby_finger','left_palm', 'left_thumb_4', 'left_thumb_3',
                        'left_thumb_2', 'left_thumb', 'left_index_finger_4', 'left_index_finger_3',
                        'left_index_finger_2', 'left_index_finger', 'left_middle_finger_4', 'left_middle_finger_3',
                        'left_middle_finger_2', 'left_middle_finger', 'left_ring_finger_4', 'left_ring_finger_3',
                        'left_ring_finger_2', 'left_ring_finger', 'left_baby_finger_4', 'left_baby_finger_3',
                        'left_baby_finger_2', 'left_baby_finger']

right_keypoint_names = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle',
                        'right_eye', 'right_ear', 'right_big_toe', 'right_small_toe', 'right_heel', 'right_palm',
                        'right_thumb_4', 'right_thumb_3', 'right_thumb_2', 'right_thumb', 'right_index_finger_4',
                        'right_index_finger_3', 'right_index_finger_2', 'right_index_finger', 'right_middle_finger_4',
                        'right_middle_finger_3', 'right_middle_finger_2', 'right_middle_finger', 'right_ring_finger_4',
                        'right_ring_finger_3', 'right_ring_finger_2', 'right_ring_finger', 'right_baby_finger_4',
                        'right_baby_finger_3', 'right_baby_finger_2', 'right_baby_finger']
left_keypoint_names = ['left_shoulder', 'left_elbow', 'left_wrist', 'left_hip', 'left_knee', 'left_ankle',
                       'left_eye', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'left_palm',
                       'left_thumb_4', 'left_thumb_3', 'left_thumb_2', 'left_thumb', 'left_index_finger_4',
                       'left_index_finger_3', 'left_index_finger_2', 'left_index_finger', 'left_middle_finger_4',
                       'left_middle_finger_3', 'left_middle_finger_2', 'left_middle_finger', 'left_ring_finger_4',
                       'left_ring_finger_3', 'left_ring_finger_2', 'left_ring_finger', 'left_baby_finger_4',
                       'left_baby_finger_3', 'left_baby_finger_2', 'left_baby_finger']
shared_keypoint_names = ['nose', 'neck', 'MidHip']
first_frame_depth_neighbours = {'neck': ['nose', 'right_eye', 'right_ear', 'left_eye', 'left_ear', 'right_shoulder', 'left_shoulder'], 
                                'right_shoulder': ['right_elbow', 'right_wrist','right_palm','right_thumb_4', 'right_thumb_3', 'right_thumb_2', 'right_thumb', 'right_index_finger_4', 'right_index_finger_3', 'right_index_finger_2', 'right_index_finger', 'right_middle_finger_4', 'right_middle_finger_3', 'right_middle_finger_2', 'right_middle_finger', 'right_ring_finger_4', 'right_ring_finger_3', 'right_ring_finger_2', 'right_ring_finger', 'right_baby_finger_4', 'right_baby_finger_3', 'right_baby_finger_2', 'right_baby_finger'],
                                'left_shoulder': ['left_elbow', 'left_wrist', 'left_palm', 'left_thumb_4', 'left_thumb_3', 'left_thumb_2', 'left_thumb', 'left_index_finger_4', 'left_index_finger_3', 'left_index_finger_2', 'left_index_finger', 'left_middle_finger_4', 'left_middle_finger_3', 'left_middle_finger_2', 'left_middle_finger', 'left_ring_finger_4', 'left_ring_finger_3', 'left_ring_finger_2', 'left_ring_finger', 'left_baby_finger_4', 'left_baby_finger_3', 'left_baby_finger_2', 'left_baby_finger'],
                                }

def get_xyz_coordinates(current_frame_coordinates, previous_frame_coordinates, video_height):
    xyz_coordinates = {}
    if previous_frame_coordinates is None: # Current Frame is First, z-cordinate will b zero
        current_front_keypoints = current_frame_coordinates['front']
        current_right_keypoints = current_frame_coordinates['right']
        current_left_keypoints = current_frame_coordinates['left']
        for keypoint_name, xy_coordinate in current_front_keypoints.items():
            if xy_coordinate is not None:
                xyz_coordinates[keypoint_name] = [xy_coordinate[0], video_height-xy_coordinate[1], 0.0]
            else:
                xyz_coordinates[keypoint_name] = None
    else:
        xyz_coordinates_unordered = {}
        current_front_keypoints = current_frame_coordinates['front']
        current_right_keypoints = current_frame_coordinates['right']
        current_left_keypoints = current_frame_coordinates['left']
        previous_front_keypoints = previous_frame_coordinates['front']
        previous_right_keypoints = previous_frame_coordinates['right']
        previous_left_keypoints = previous_frame_coordinates['left']
        for keypoint_name in right_keypoint_names:
            if current_front_keypoints[keypoint_name] is not None:
                if current_right_keypoints[keypoint_name] is not None and previous_right_keypoints[keypoint_name] is not None:
                    z_coordinate = current_right_keypoints[keypoint_name][0] - previous_right_keypoints[keypoint_name][0]
                    xyz_coordinates_unordered[keypoint_name] = [*current_front_keypoints[keypoint_name], z_coordinate]
                else:
                    xyz_coordinates_unordered[keypoint_name] = [*current_front_keypoints[keypoint_name], 0.0]
            else:
                xyz_coordinates_unordered[keypoint_name] = [*previous_front_keypoints[keypoint_name], 0.0]
        for keypoint_name in left_keypoint_names:
            if current_front_keypoints[keypoint_name] is not None:
                if current_left_keypoints[keypoint_name] is not None and previous_left_keypoints[keypoint_name] is not None:
                    z_coordinate = current_left_keypoints[keypoint_name][0] - previous_left_keypoints[keypoint_name][0]
                    xyz_coordinates_unordered[keypoint_name] = [*current_front_keypoints[keypoint_name], z_coordinate]
                else:
                    xyz_coordinates_unordered[keypoint_name] = [*current_front_keypoints[keypoint_name], 0.0]
            else:
                xyz_coordinates_unordered[keypoint_name] = [*previous_front_keypoints[keypoint_name], 0.0]
        for keypoint_name in shared_keypoint_names:
            if current_front_keypoints[keypoint_name] is not None:
                if current_right_keypoints[keypoint_name] is not None and previous_right_keypoints[keypoint_name] is not None:
                    z_coordinate = current_right_keypoints[keypoint_name][0] - previous_right_keypoints[keypoint_name][0]
                    xyz_coordinates_unordered[keypoint_name] = (*current_front_keypoints[keypoint_name], z_coordinate)
                elif current_left_keypoints[keypoint_name] is not None and previous_left_keypoints[keypoint_name] is not None:
                    z_coordinate = current_left_keypoints[keypoint_name][0] - previous_left_keypoints[keypoint_name][0]
                    xyz_coordinates_unordered[keypoint_name] = [*current_front_keypoints[keypoint_name], z_coordinate]
                else:
                    xyz_coordinates_unordered[keypoint_name] = [*current_front_keypoints[keypoint_name], 0.0]
            else:
                xyz_coordinates_unordered[keypoint_name] = [*previous_front_keypoints[keypoint_name], 0.0]
        for ordered_keypoint_name in ordered_keypoint_names:
            xyz_coordinates[ordered_keypoint_name] = xyz_coordinates_unordered[ordered_keypoint_name]
    return xyz_coordinates

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
