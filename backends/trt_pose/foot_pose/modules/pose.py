import cv2
from backends.trt_pose.foot_pose.modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS

class Pose:
    num_kpts = 24
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear', 'l_big_toe', 'l_small_toe', 'l_heel', 'r_big_toe', 'r_small_toe', 'r_heel']
    feet_keypoint_ids = [10, 13, 18, 19, 20, 21, 22, 23] 
    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence

    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)
        for part_id in range(len(BODY_PARTS_PAF_IDS)):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 5, (0, 255, 0), -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 5, (255, 0, 0), -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (0, 0, 255), 4)

    def parse_feet_keypoints(self):
        assert self.keypoints.shape == (Pose.num_kpts, 2)
        feet_keypoints = []
        for keypoint_id in self.feet_keypoint_ids:
            if self.keypoints[keypoint_id, 0] != -1:
                x, y = self.keypoints[keypoint_id]
                feet_keypoints.append((x, y))
            else:
                feet_keypoints.append(None)
        return feet_keypoints

