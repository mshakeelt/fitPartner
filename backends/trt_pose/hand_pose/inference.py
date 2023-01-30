import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
import numpy as np
import traitlets
import torch
import torchvision.transforms as transforms
import PIL.Image
import trt_pose.coco
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

hand_pose_path = os.path.join("hand_pose", "preprocess", "hand_pose.json")
hand_model_weights =  os.path.join("hand_pose", "trained_models", "hand_pose_resnet18_att_244_244.pth")
with open(hand_pose_path, 'r') as f:
    hand_pose = json.load(f)

hand_topology = trt_pose.coco.coco_category_to_topology(hand_pose)


num_hand_parts = len(hand_pose['keypoints'])
num_hand_links = len(hand_pose['skeleton'])
hand_keypoint_names = hand_pose['keypoints']
hand_model = trt_pose.models.resnet18_baseline_att(num_hand_parts, 2 * num_hand_links).cuda().eval()
hand_model.load_state_dict(torch.load(hand_model_weights))

WIDTH = 224
HEIGHT = 224
parse_hand_objects = ParseObjects(hand_topology,cmap_threshold=0.15, link_threshold=0.15)
draw_hand_objects = DrawObjects(hand_topology)

hand_model_target_shape = (224, 224)
hand_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
hand_std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image).resize(hand_model_target_shape)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(hand_mean[:, None, None]).div_(hand_std[:, None, None])
    return image[None, ...]


def execute(image):
    data = preprocess(image)
    cmap, paf = hand_model(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    count, objects, peaks = parse_hand_objects(cmap, paf)
    hands_keypoint_coordinates = get_positions(image, count, objects, peaks)
    return hands_keypoint_coordinates


def get_positions(image, count, objects, normalized_peaks):
        height = image.shape[0]
        width = image.shape[1]
        both_hands_positions = {}
        print(count)
        count = count[0] if count[0]<3 else 2
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
                    per_hand_positions[hand_keypoint_names[j]] = (x, y)
                else:
                    per_hand_positions[hand_keypoint_names[j]] = None
            both_hands_positions[f"hand_{i}"] = per_hand_positions
        return both_hands_positions

image_path = r"test_media\7.jpg"
img = cv2.imread(image_path)
img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
hands_keypoint_coordinates = execute(image=img)
print(hands_keypoint_coordinates)