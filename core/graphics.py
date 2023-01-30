import cv2

class Graphics(object):
    def __init__(self, colors, comparison_level, hands=True, feet=True, show_full_hands=False):
        self.colors = colors
        if comparison_level.lower() == 'full':
            self.skeleton_points = [("left_hip", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"), 
                                    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"), 
                                    ("left_eye", "right_eye"), ("left_eye", "nose"), ("right_eye", "nose"), ("left_eye", "left_ear"), ("right_eye", "right_ear"), ("left_ear", "left_shoulder"), 
                                    ("right_ear", "right_shoulder"), ("nose", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"), ("neck", "left_hip"), 
                                    ("neck", "right_hip")]
        elif comparison_level.lower() == 'upper':
            self.skeleton_points = [("left_hip", "right_hip"), ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"), 
                                    ("right_elbow", "right_wrist"), ("left_eye", "right_eye"), ("left_eye", "nose"), ("right_eye", "nose"), ("left_eye", "left_ear"), 
                                    ("right_eye", "right_ear"), ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"), ("nose", "neck"), ("neck", "left_shoulder"), 
                                    ("neck", "right_shoulder"), ("neck", "left_hip"), ("neck", "right_hip")]
        elif comparison_level.lower() == 'lower':
            self.skeleton_points = [("left_hip", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")]
        
        if comparison_level.lower() == 'full' or comparison_level.lower() == 'lower':
            if feet:
                self.skeleton_points.extend([("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"), ("left_ankle", "left_heel") , ("right_ankle", "right_big_toe"), 
                                            ("right_ankle", "right_small_toe"), ("right_ankle", "right_heel")])
        if comparison_level.lower() == 'full' or comparison_level.lower() == 'upper':
            if hands:
                if show_full_hands:
                    self.skeleton_points.extend([("right_wrist", "right_thumb_4"), ('right_thumb_4', "right_thumb_3"), ('right_thumb_3', "right_thumb_2"), ('right_thumb_2' , "right_thumb"), 
                                                ("right_wrist", "right_index_finger_4"), ("right_index_finger_4", "right_index_finger_3"), ("right_index_finger_3", "right_index_finger_2"), ("right_index_finger_2", "right_index_finger"), 
                                                ("right_wrist", "right_middle_finger_4"), ("right_middle_finger_4", "right_middle_finger_3"), ("right_middle_finger_3", "right_middle_finger_2"), ("right_middle_finger_2", "right_middle_finger"),
                                                ("right_wrist", "right_ring_finger_4"), ("right_ring_finger_4", "right_ring_finger_3"), ("right_ring_finger_3", "right_ring_finger_2"), ("right_ring_finger_2", "right_ring_finger"),
                                                ("right_wrist", "right_baby_finger_4"), ("right_baby_finger_4", "right_baby_finger_3"), ("right_baby_finger_3", "right_baby_finger_2"), ("right_baby_finger_2", "right_baby_finger"),
                                                ("left_wrist", "left_thumb_4"), ('left_thumb_4', "left_thumb_3"), ('left_thumb_3', "left_thumb_2"), ('left_thumb_2' , "left_thumb"), 
                                                ("left_wrist", "left_index_finger_4"), ("left_index_finger_4", "left_index_finger_3"), ("left_index_finger_3", "left_index_finger_2"), ("left_index_finger_2", "left_index_finger"), 
                                                ("left_wrist", "left_middle_finger_4"), ("left_middle_finger_4", "left_middle_finger_3"), ("left_middle_finger_3", "left_middle_finger_2"), ("left_middle_finger_2", "left_middle_finger"),
                                                ("left_wrist", "left_ring_finger_4"), ("left_ring_finger_4", "left_ring_finger_3"), ("left_ring_finger_3", "left_ring_finger_2"), ("left_ring_finger_2", "left_ring_finger"),
                                                ("left_wrist", "left_baby_finger_4"), ("left_baby_finger_4", "left_baby_finger_3"), ("left_baby_finger_3", "left_baby_finger_2"), ("left_baby_finger_2", "left_baby_finger")])
                else:
                    self.skeleton_points.extend([("right_wrist", "right_thumb"), ("right_wrist", "right_index_finger"), ("right_wrist", "right_middle_finger"), ("right_wrist", "right_ring_finger"), 
                                                ("right_wrist", "right_baby_finger"), ("left_wrist", "left_thumb"), ("left_wrist", "left_index_finger"), ("left_wrist", "left_middle_finger"), 
                                                ("left_wrist", "left_ring_finger"), ("left_wrist", "left_baby_finger")])

    def __call__(self, image, keypoints, subject=None, keypoint_comparison_results=None, joints_comparison_results=None):
        if subject.lower() == 'master':
            for points in self.skeleton_points:
                if keypoints[points[0]] is not None and keypoints[points[1]] is not None:                    
                    x0, y0 = keypoints[points[0]]
                    x1, y1 = keypoints[points[1]]
                    cv2.circle(image, (round(x0), round(y0)), 3, self.colors['master_keypoint_color'], 2)
                    cv2.circle(image, (round(x1), round(y1)), 3, self.colors['master_keypoint_color'], 2)
                    cv2.line(image, (round(x0), round(y0)), (round(x1), round(y1)), self.colors['master_skeleton_color'], 2)
        elif subject.lower() == 'follower':
            for points in self.skeleton_points:
                if keypoints[points[0]] is not None and keypoints[points[1]] is not None:
                    x0, y0 = keypoints[points[0]]
                    x1, y1 = keypoints[points[1]]
                    if keypoint_comparison_results[points[0]]:
                        cv2.circle(image, (round(x0), round(y0)), 3, self.colors['follower_keypoint_match_color'], 2)
                    else:
                        cv2.circle(image, (round(x0), round(y0)), 3, self.colors['follower_keypoint_mismatch_color'], 2)
                    if keypoint_comparison_results[points[1]]:
                        cv2.circle(image, (round(x1), round(y1)), 3, self.colors['follower_keypoint_match_color'], 2)
                    else:
                        cv2.circle(image, (round(x1), round(y1)), 3, self.colors['follower_keypoint_mismatch_color'], 2)
                    if joints_comparison_results[points] is not None:
                        if joints_comparison_results[points]:
                            cv2.line(image, (round(x0), round(y0)), (round(x1), round(y1)), self.colors['follower_skeleton_match_color'], 2)
                        else:
                            cv2.line(image, (round(x0), round(y0)), (round(x1), round(y1)), self.colors['follower_skeleton_mismatch_color'], 2)