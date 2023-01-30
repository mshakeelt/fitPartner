import core.inference as inference
from core.inference import TRT_Infer_Body, TRT_Infer_Feet, TRT_Infer_Hands
from core.comparison import Compare
from core.signaling import Signals
from core.graphics import Graphics
from threading import Thread
from queue import Queue
import cv2
import os

queue_out = Queue()
def draw_inference(master_image, follower_image):
    trt_inference_results = {}
    t1 = Thread(target=trt_infer_body.execute_with_queue, args=(queue_out, 'master', master_image))
    t2 = Thread(target=trt_infer_body.execute_with_queue, args=(queue_out, 'follower', follower_image))
    t1.start()
    t2.start()
    if body_part.lower() == 'full':
        t3 = Thread(target=trt_infer_hands.execute_with_queue, args=(queue_out, 'master', master_image))
        t4 = Thread(target=trt_infer_hands.execute_with_queue, args=(queue_out, 'follower', follower_image))
        t5 = Thread(target=trt_infer_feet.execute_with_queue, args=(queue_out, 'master', master_image))
        t6 = Thread(target=trt_infer_feet.execute_with_queue, args=(queue_out, 'follower', follower_image))
        t3.start()
        t4.start()
        t5.start()
        t6.start()
    elif body_part.lower() == 'upper':
        trt_inference_results['master_feet'] = None
        trt_inference_results['follower_feet'] = None
        t3 = Thread(target=trt_infer_hands.execute_with_queue, args=(queue_out, 'master', master_image))
        t4 = Thread(target=trt_infer_hands.execute_with_queue, args=(queue_out, 'follower', follower_image))
        t3.start()
        t4.start()
    elif body_part.lower() == 'lower':
        trt_inference_results['master_hands'] = None
        trt_inference_results['follower_hands'] = None
        t3 = Thread(target=trt_infer_feet.execute_with_queue, args=(queue_out, 'master', master_image))
        t4 = Thread(target=trt_infer_feet.execute_with_queue, args=(queue_out, 'follower', follower_image))
        t3.start()
        t4.start()
    t1.join()
    t2.join()
    if body_part.lower() == 'full':
        t3.join()
        t4.join()
        t5.join()
        t6.join()
    elif body_part.lower() == 'upper' or body_part.lower() == 'lower':
        t3.join()
        t4.join()

    while not queue_out.empty():
        trt_inference_results.update(queue_out.get())

    t7 = Thread(target=inference.join_trt_keypoints_with_queue, args=(queue_out, 'master', trt_inference_results['master_body'], trt_inference_results['master_hands'], trt_inference_results['master_feet']))
    t8 = Thread(target=inference.join_trt_keypoints_with_queue, args=(queue_out, 'follower', trt_inference_results['follower_body'], trt_inference_results['follower_hands'], trt_inference_results['follower_feet']))
    t7.start()
    t8.start()
    t7.join()
    t8.join()

    trt_joining_results = {}
    while not queue_out.empty():
        trt_joining_results.update(queue_out.get())
    
    trt_master_keypoints = trt_joining_results['master']
    trt_follower_keypoints = trt_joining_results['follower'] 
    trt_keypoint_comparison_results, trt_joints_comparison_results, trt_angle_errors = draw_comparison.compare(trt_master_keypoints, trt_follower_keypoints)
    
    if output_graphics:
        t9 = Thread(target=generate_signals, args=(trt_joints_comparison_results, trt_angle_errors))
        t10 = Thread(target=draw_graphics, args=(master_image, trt_master_keypoints, 'master'))
        t11 = Thread(target=draw_graphics, args=(follower_image, trt_follower_keypoints, 'follower', trt_keypoint_comparison_results, trt_joints_comparison_results))
        t9.start()
        t10.start()
        t11.start()
        t9.join()
        t10.join()
        t11.join()
    else:
        generate_signals(trt_joints_comparison_results, trt_angle_errors)
    return master_image, follower_image

def process_videos(master_video_path):
    master_cap = cv2.VideoCapture(master_video_path)
    follower_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    comparison_frames_available = True
    print("Comparison started. Press q to quit! \n")
    while(comparison_frames_available):
        master_ret, master_frame = master_cap.read()
        follower_ret, follower_frame = follower_cap.read()
        comparison_frames_available = master_ret and follower_ret
        if comparison_frames_available:
            infered_master_image, infered_follower_image = draw_inference(master_frame, follower_frame)
            infered_master_image = cv2.resize(infered_master_image, (640, 480), interpolation = cv2.INTER_AREA)
            infered_follower_image = cv2.resize(infered_follower_image, (640, 480), interpolation = cv2.INTER_AREA)
            if output_graphics:
                cv2.imshow('Master', infered_master_image)
                cv2.imshow('Follower', infered_follower_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting!")
                break
        elif not master_ret and follower_ret:
            print("Master Video is Finished. Exitting...")
        elif not follower_ret and master_ret:
            print("Follower Video is Finished. Exitting...")
        else:
            print("Both videos are finished!")

    master_cap.release()
    follower_cap.release()
    if output_graphics:
        cv2.destroyAllWindows()

def process_images(master_frame_path, follower_frame_path):
    master_frame = cv2.imread(master_frame_path)
    follower_frame = cv2.imread(follower_frame_path)
    infered_master_image, infered_follower_image = draw_inference(master_frame, follower_frame)
    infered_master_image = cv2.resize(infered_master_image, (640, 480), interpolation = cv2.INTER_AREA)
    infered_follower_image = cv2.resize(infered_follower_image, (640, 480), interpolation = cv2.INTER_AREA)
    if output_graphics:
        cv2.imshow('Master', infered_master_image)
        cv2.imshow('Follower', infered_follower_image)
        cv2.waitKey(0)
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
    threshold = 15 #angle difference in degrees for comparison
    output_graphics = True 
    if output_graphics:
        colors = set_colors()
        draw_graphics = Graphics(colors=colors, comparison_level=body_part.lower())
    
    file_name = "comparison_results.txt"
    completeName = os.path.join("output", file_name)
    output_file = open(completeName, "w+")
    output_file.write("Comparison started!")
    output_file.write("\n")
    
    draw_comparison = Compare(comparison_level=body_part, threshold=threshold)
    generate_signals = Signals(file_to_write=output_file)        
    
    trt_infer_body = TRT_Infer_Body()
    if body_part.lower() == 'full':
        trt_infer_hands = TRT_Infer_Hands()
        trt_infer_feet = TRT_Infer_Feet()
    elif body_part.lower() == 'upper':
        trt_infer_hands = TRT_Infer_Hands()
    elif body_part.lower() == 'lower':
        trt_infer_feet = TRT_Infer_Feet()
        
    master_video_path = os.path.join("test_media", "exercise.mp4")
    process_videos(master_video_path)
    output_file.close()