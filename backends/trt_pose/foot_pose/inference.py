import cv2
import numpy as np
import torch
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose
from modules.val import normalize, pad_width
from modules.with_mobilenet import PoseEstimationWithMobileNet

def preprocess(image):
    height, _, _ = image.shape
    scale = net_input_height_size / height
    scaled_img = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    tensor_img = tensor_img.cuda()
    return tensor_img, pad, scale

def post_process(heatmaps, pafs, pad, scale):
    total_keypoints_num = 0
    all_keypoints_by_type = []
    num_keypoints = Pose.num_kpts
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
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

def draw_indeference(image, model):
    model = model.eval().cuda()
    tensor_img, pad, scale = preprocess(image=image)
    stages_output = model(tensor_img)
    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    current_poses = post_process(heatmaps=heatmaps, pafs=pafs, pad=pad, scale=scale)
    feet_keypoints = [None] * 8
    for pose in current_poses:
        pose.draw(image)
        foot_keypoints = pose.parse_feet_keypoints()
        for index, foot_keypoint in enumerate(foot_keypoints):
            if foot_keypoint:
                feet_keypoints[index] = foot_keypoint

    print("Locations: ", feet_keypoints)
    cv2.imshow('just img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

upsample_ratio = 4
net_input_height_size = 256
pad_value=(0, 0, 0)
img_mean=(128, 128, 128)
img_scale=1/256
stride = 8

image_path = r"test_media\3.jpg"
checkpoint_path = r'trained_models\checkpoint_iter_384000.pth'

frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

network = PoseEstimationWithMobileNet(num_heatmaps=25, num_pafs=50)
checkpoint = torch.load(checkpoint_path)
load_state(network, checkpoint)

keypoint_locations = draw_indeference(frame, network)