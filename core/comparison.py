from scipy.spatial.distance import euclidean
import math

class Compare(object):
    def __init__(self, comparison_level, threshold) -> None:
        self.comparison_level = comparison_level
        self.threshold = threshold
        if self.comparison_level.lower() == 'full':
            self.skeleton_points = [("left_hip", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_big_toe"), 
                                    ("left_ankle", "left_small_toe"), ("left_ankle", "left_heel"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"), 
                                    ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"), ("right_ankle", "right_heel"), ("left_shoulder", "left_elbow"), 
                                    ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"), ("left_eye", "right_eye"), 
                                    ("left_eye", "nose"), ("right_eye", "nose"), ("left_eye", "left_ear"), ("right_eye", "right_ear"), ("left_ear", "left_shoulder"), 
                                    ("right_ear", "right_shoulder"), ("nose", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"), ("neck", "left_hip"), 
                                    ("neck", "right_hip"), ("right_wrist", "right_thumb"), ("right_wrist", "right_index_finger"), ("right_wrist", "right_middle_finger"), 
                                    ("right_wrist", "right_ring_finger"), ("right_wrist", "right_baby_finger"), ("left_wrist", "left_thumb"), ("left_wrist", "left_index_finger"), 
                                    ("left_wrist", "left_middle_finger"), ("left_wrist", "left_ring_finger"), ("left_wrist", "left_baby_finger")]
        elif self.comparison_level.lower() == 'upper':
            self.skeleton_points = [("left_hip", "right_hip"), ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"), 
                                    ("right_elbow", "right_wrist"), ("left_eye", "right_eye"), ("left_eye", "nose"), ("right_eye", "nose"), ("left_eye", "left_ear"), 
                                    ("right_eye", "right_ear"), ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"), ("nose", "neck"), ("neck", "left_shoulder"), 
                                    ("neck", "right_shoulder"), ("neck", "left_hip"), ("neck", "right_hip"), ("right_wrist", "right_thumb"), ("right_wrist", "right_index_finger"), 
                                    ("right_wrist", "right_middle_finger"), ("right_wrist", "right_ring_finger"), ("right_wrist", "right_baby_finger"), ("left_wrist", "left_thumb"), 
                                    ("left_wrist", "left_index_finger"), ("left_wrist", "left_middle_finger"), ("left_wrist", "left_ring_finger"), ("left_wrist", "left_baby_finger")]
        elif self.comparison_level.lower() == 'lower':
            self.skeleton_points = [("left_hip", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_big_toe"), 
                                    ("left_ankle", "left_small_toe"), ("left_ankle", "left_heel") ,("right_hip", "right_knee"), ("right_knee", "right_ankle"), 
                                    ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"), ("right_ankle", "right_heel")]

    def compare(self, master_keypoints, follower_keypoints):
        self.master_keypoints = master_keypoints
        self.follower_keypoints = follower_keypoints
        keypoint_comparison_results, joints_comparison_results, angle_errors = self.compare_angles()
        return keypoint_comparison_results, joints_comparison_results, angle_errors

    def get_anchors(self):
        self.top_anchor = None
        self.bottom_anchor = None
        if self.comparison_level.lower() == 'full':
            top_anchors_list = [0, 1, 2, 3, 4, 17, 5, 6]
            bottom_anchors_list = [-2, -3, -4, -5, -6, -7]
        elif self.comparison_level.lower() == 'upper':
            top_anchors_list = [0, 1, 2, 3, 4, 13, 5, 6]
            bottom_anchors_list = [-2, -3, -4, -5, -6, -7]
        elif self.comparison_level.lower() == 'lower':
            top_anchors_list = [0, 1]
            bottom_anchors_list = [-1, -2]
        for ii in top_anchors_list:
            if all(self.master_keypoints[ii]) and all(self.follower_keypoints[ii]):
                self.top_anchor = ii
                break
        for jj in bottom_anchors_list:
            if all(self.master_keypoints[jj]) and all(self.follower_keypoints[jj]):
                self.bottom_anchor = jj
                break
        
    def set_ref_error(self):
        master_ref_joint = euclidean(self.master_keypoints[self.top_anchor], self.master_keypoints[self.bottom_anchor])
        follower_ref_joint = euclidean(self.follower_keypoints[self.top_anchor], self.follower_keypoints[self.bottom_anchor])
        self.ref_error = master_ref_joint - follower_ref_joint
        print("Reference Error: ", self.ref_error)

    def compare_keypoints(self):
        comparison_results = []
        master_ref_joint = euclidean(self.master_keypoints[self.top_anchor], self.master_keypoints[self.bottom_anchor])
        follower_ref_joint = euclidean(self.follower_keypoints[self.top_anchor], self.follower_keypoints[self.bottom_anchor])
        for index in range(len(self.master_keypoints)):
            if all(self.master_keypoints[index]) and all(self.follower_keypoints[index]):
                master_sec_joint = euclidean(self.master_keypoints[self.top_anchor], self.master_keypoints[index])
                follower_sec_joint = euclidean(self.follower_keypoints[self.top_anchor], self.follower_keypoints[index])
                if master_sec_joint==0:
                    master_ratio = 0
                else:
                    master_ratio = master_ref_joint/master_sec_joint
                if follower_sec_joint==0:
                    follower_ratio = 0
                else:    
                    follower_ratio = follower_ref_joint/follower_sec_joint
                joint_error = master_ratio - follower_ratio
                if joint_error > self.ref_error:
                    comparison_results.append(False)
                else:
                    comparison_results.append(True)
            else:
                comparison_results.append(False)
        return comparison_results

    def get_angle(self, pointA, pointB):
        dy = pointA[1]-pointB[1]
        dx = pointA[0]-pointB[0]
        if dy<0 and dx>0:
            angle_in_radians = math.atan2(abs(dy), dx)
            angle_in_degrees = math.degrees(angle_in_radians)
        elif dy<0 and dx<0:
            angle_in_radians = math.atan2(abs(dy), abs(dx))
            angle_in_degrees = 180-math.degrees(angle_in_radians)
        elif dy>=0 and dx<0:
            angle_in_radians = math.atan2(dy, abs(dx))
            angle_in_degrees = 180+math.degrees(angle_in_radians)
        elif dy>=0 and dx>0:
            angle_in_radians = math.atan2(dy, dx)
            angle_in_degrees = -math.degrees(angle_in_radians)
        elif dy>0 and dx==0:
            angle_in_degrees = 270
        elif dy<0 and dx==0:
            angle_in_degrees = 90
        elif dy==0 and dx==0:
            angle_in_degrees = 0
        angle_in_degrees = 360-angle_in_degrees if angle_in_degrees>=360 else angle_in_degrees
        return angle_in_degrees

    def compare_angles(self):
        joints_comparison_results = dict.fromkeys(self.skeleton_points, None)
        angle_errors = dict.fromkeys(self.skeleton_points, None)
        keypoint_comparison_results = dict.fromkeys(self.master_keypoints, True)
        master_angles = []
        follower_angles = []
        for points in self.skeleton_points:
            if self.master_keypoints[points[0]] is not None and self.master_keypoints[points[1]] is not None:
                master_joint_angle = self.get_angle(self.master_keypoints[points[0]], self.master_keypoints[points[1]])
            else:
                master_joint_angle = None
            if self.follower_keypoints[points[0]] is not None and self.follower_keypoints[points[1]] is not None:
                follower_joint_angle = self.get_angle(self.follower_keypoints[points[0]], self.follower_keypoints[points[1]])
            else:
                follower_joint_angle = None
            if master_joint_angle is not None and follower_joint_angle is not None:
                error = master_joint_angle-follower_joint_angle
                angle_errors[points] = round(error, 2)
                if abs(error)<self.threshold:
                    joints_comparison_results[points] = True
                else:
                    joints_comparison_results[points] = False
                    keypoint_comparison_results[points[0]] = False
                    keypoint_comparison_results[points[1]] = False
            elif master_joint_angle is None and follower_joint_angle is None:
                keypoint_comparison_results[points[0]] = False
                keypoint_comparison_results[points[1]] = False
            else:
                joints_comparison_results[points] = False
                keypoint_comparison_results[points[0]] = False
                keypoint_comparison_results[points[1]] = False
            master_angles.append(master_joint_angle)
            follower_angles.append(follower_joint_angle)
        """ print("master_angles: ", len(master_angles))
        print("follower_angles: ", len(follower_angles))
        print("keypoint_comparison_results: ", len(keypoint_comparison_results)) 
        print("joints_comparison_results: ", joints_comparison_results, '\n \n') 
        print("angle_errors: ", angle_errors)
        print("\n") """
        return keypoint_comparison_results, joints_comparison_results, angle_errors
