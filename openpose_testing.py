import cv2
from core.inference import Open_Pose_Infer_Full
from core.graphics import Graphics


def set_colors():
    colors = {}
    colors['master_skeleton_color'] = (255, 255, 255)
    colors['master_keypoint_color'] = (128, 128, 128)
    colors['follower_skeleton_match_color'] = (74, 225, 180)
    colors['follower_keypoint_match_color'] = (74, 225, 180)
    colors['follower_skeleton_mismatch_color'] = (80, 97, 230)
    colors['follower_keypoint_mismatch_color'] = (80, 97, 230)
    return colors

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    print("Demo started. Press q to quit! \n")
    while(True):
        ret, frame = cap.read()
        if ret:
            full_keypoints = open_pose_infer_full.execute(frame)
            draw_graphics(frame, full_keypoints, 'master')
            frame = cv2.resize(frame, (480, 720), interpolation = cv2.INTER_AREA)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting!")
                break
        else:
            print("Video is finished!")
            break
    cap.release()
    cv2.destroyAllWindows()

body_part = 'full' # full, upper, or lower
show_hands = True
show_full_hands = True
show_feet = True
source = 'cam' # 'file' or 'cam'
colors = set_colors()
draw_graphics = Graphics(colors=colors, comparison_level=body_part.lower(), hands=show_hands, feet=show_feet, show_full_hands=show_full_hands)        

open_pose_infer_full = Open_Pose_Infer_Full()
video_path = r"test_media\vid1.mp4"
process_video(video_path)

"""
body25
{0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle', 15: 'REye', 16: 'LEye', 17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 
22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel', 25: 'Background'}
hand21
left: hand0
right: hand1
{0: 'palm', 1: 'thumb_2', 2: 'thumb_3', 3: 'thumb_4', 4: 'thumb', 5: 'index_finger_2', 6: 'index_finger_3', 7: 'index_finger_4', 8: 'index_finger',
 9: 'middle_finger_2', 10: 'middle_finger_3', 11: 'middle_finger_4', 12: 'middle_finger', 13: 'ring_finger_2', 14: 'ring_finger_3', 15: 'ring_finger_4', 16: 'ring_finger',
 17: 'baby_finger_2', 18: 'baby_finger_3', 19: 'baby_finger_4', 20: 'baby_finger'}
"""