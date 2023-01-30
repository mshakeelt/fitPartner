import core.inference as inference
from core.inference import Open_Pose_Infer_Full
from core.graphics import Graphics
import cv2

def draw_inference(front_image, right_image, left_image):
    front_keypoints = open_pose_infer_full.execute(front_image)
    right_keypoints = open_pose_infer_full.execute(right_image)
    left_keypoints = open_pose_infer_full.execute(left_image)
    draw_graphics(front_image, front_keypoints, 'master')
    draw_graphics(right_image, right_keypoints, 'master')
    draw_graphics(left_image, left_keypoints, 'master')
    three_cam_coordinates = {'front': front_keypoints, 'right': right_keypoints, 'left': left_keypoints}
    return front_image, right_image, left_image, three_cam_coordinates

def process_cams():
    front_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    right_cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    left_cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    print("Recording started. Press q to quit! \n")
    previous_frame_coordinates = None
    while(True):
        front_ret, front_frame = front_cam.read()
        right_ret, right_frame = right_cam.read()
        left_ret, left_frame = left_cam.read()
        if front_ret and right_ret and left_ret:
            infered_front_frame, infered_right_frame, infered_left_frame, current_frame_coordinates = draw_inference(front_frame, right_frame, left_frame)
            cv2.imshow('Front Camera Stream', infered_front_frame)
            cv2.imshow('Right Camera Stream', infered_right_frame)
            cv2.imshow('Left Camera Stream', infered_left_frame)
            xyz_coordinates = inference.get_xyz_coordinates(current_frame_coordinates, previous_frame_coordinates)
            previous_frame_coordinates = current_frame_coordinates
            print(xyz_coordinates)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting!")
                break
        else:
            print("Video is finished!")
            break
    front_cam.release()
    right_cam.release()
    left_cam.release()
    cv2.destroyAllWindows()

def test_cams():
    front_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    right_cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    left_cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    print("camera testing started. Press q to quit! \n")
    while(True):
        front_ret, front_frame = front_cam.read()
        right_ret, right_frame = right_cam.read()
        left_ret, left_frame = left_cam.read()
        if front_ret and right_ret and left_ret:
            cv2.imshow('Front Camera Stream', front_frame)
            cv2.imshow('Right Camera Stream', right_frame)
            cv2.imshow('Left Camera Stream', left_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting!")
                break
        else:
            print("Cameras are not accessable!")
            print("Exitting!")
            break
    front_cam.release()
    right_cam.release()
    left_cam.release()
    cv2.destroyAllWindows()

def set_colors():
    colors = {}
    colors['master_skeleton_color'] = (255, 255, 255)
    colors['master_keypoint_color'] = (128, 128, 128)
    colors['follower_skeleton_match_color'] = (74, 225, 180)
    colors['follower_keypoint_match_color'] = (74, 225, 180)
    colors['follower_skeleton_mismatch_color'] = (80, 97, 230)
    colors['follower_keypoint_mismatch_color'] = (80, 97, 230)
    return colors

if __name__ == '__main__':
    body_part = 'full' # full, upper, or lower
    show_hands = True
    show_full_hands = True
    show_feet = True
    colors = set_colors()
    draw_graphics = Graphics(colors=colors, comparison_level=body_part.lower(), hands=show_hands, feet=show_feet, show_full_hands=show_full_hands)
    open_pose_infer_full = Open_Pose_Infer_Full()
    test_cams()
    #process_cams()