import core.inference as inference
from core.inference import TRT_Infer_Body, TRT_Infer_Feet, TRT_Infer_Hands
from core.graphics import Graphics
import cv2
import os

def draw_inference(image):
    trt_hands_keypoints = None
    trt_feet_keypoints = None
    trt_body_keypoints = trt_infer_body.execute(image)
    if body_part.lower() == 'full':
        trt_hands_keypoints = trt_infer_hands.execute(image)
        trt_feet_keypoints = trt_infer_feet.execute(image)
    elif body_part.lower() == 'upper':
        trt_hands_keypoints = trt_infer_hands.execute(image)
    elif body_part.lower() == 'lower':
        trt_feet_keypoints = trt_infer_feet.execute(image)  
    trt_keypoints = inference.join_trt_keypoints(trt_body_keypoints, trt_hands_keypoints, trt_feet_keypoints)
    draw_graphics(image, trt_keypoints, 'master')
    return image

def process_video(video_source, video_path):
    if video_source == 'file':
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Demo started. Press q to quit! \n")
    while(True):
        ret, frame = cap.read()
        if ret:
            infered_frame = draw_inference(frame)
            cv2.imshow('Frame', infered_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting!")
                break
        else:
            print("Video is finished!")
            break
    cap.release()
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
    source = 'cam' # 'file' or 'cam'
    colors = set_colors()
    draw_graphics = Graphics(colors=colors, comparison_level=body_part.lower(), hands=show_hands, feet=show_feet, show_full_hands=show_full_hands)        
    trt_infer_body = TRT_Infer_Body()
    if body_part.lower() == 'full':
        trt_infer_hands = TRT_Infer_Hands()
        trt_infer_feet = TRT_Infer_Feet()
    elif body_part.lower() == 'upper':
        trt_infer_hands = TRT_Infer_Hands()
    elif body_part.lower() == 'lower':
        trt_infer_feet = TRT_Infer_Feet()
    if source == 'file':
        video_path = os.path.join("test_media", "exercise.mp4")
    else:
        video_path = None
    process_video(source, video_path)