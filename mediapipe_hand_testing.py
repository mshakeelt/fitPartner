import cv2
from core.inference import Mediapipe_Infer_Full
from core.graphics import Graphics

def set_colors():
    colors = {}
    colors['master_skeleton_color'] = (0, 0, 255)
    colors['master_keypoint_color'] = (128, 128, 128)
    colors['follower_skeleton_match_color'] = (74, 225, 180)
    colors['follower_keypoint_match_color'] = (74, 225, 180)
    colors['follower_skeleton_mismatch_color'] = (80, 97, 230)
    colors['follower_keypoint_mismatch_color'] = (80, 97, 230)
    return colors


body_part = 'full' # full, upper, or lower
show_hands = True
show_full_hands = True
show_feet = True
colors = set_colors()
draw_graphics = Graphics(colors=colors, comparison_level=body_part.lower(), hands=show_hands, feet=show_feet, show_full_hands=show_full_hands)        
mediapipe_infer_full = Mediapipe_Infer_Full()
image_path = r"test_media\14.jpg"
imageBGR = cv2.imread(image_path)
image_height, image_width, _ = imageBGR.shape
mediapipe_infer_full.imagewidth = image_width
mediapipe_infer_full.imageheight = image_height
full_body_keypoints = mediapipe_infer_full.execute(imageBGR)
draw_graphics(imageBGR, full_body_keypoints, 'master')
cv2.imshow('Media_Pipe', imageBGR)
cv2.waitKey(0)
cv2.destroyAllWindows()
