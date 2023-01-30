import numpy as np
import os


video_keypoints = np.load('output\corrected_keypoints.npy')
print(np.max(video_keypoints[2]-video_keypoints[1]))
print(video_keypoints.shape)

first_frame_names = ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow',
                    'left_wrist', 'MidHip', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
                    'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe',
                    'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'right_palm', 'right_thumb_4',
                    'right_thumb_3', 'right_thumb_2', 'right_thumb', 'right_index_finger_4', 'right_index_finger_3',
                    'right_index_finger_2', 'right_index_finger', 'right_middle_finger_4', 'right_middle_finger_3',
                    'right_middle_finger_2', 'right_middle_finger', 'right_ring_finger_4', 'right_ring_finger_3',
                    'right_ring_finger_2', 'right_ring_finger', 'right_baby_finger_4', 'right_baby_finger_3',
                    'right_baby_finger_2', 'right_baby_finger','left_palm', 'left_thumb_4', 'left_thumb_3',
                    'left_thumb_2', 'left_thumb', 'left_index_finger_4', 'left_index_finger_3', 'left_index_finger_2',
                    'left_index_finger', 'left_middle_finger_4', 'left_middle_finger_3', 'left_middle_finger_2',
                    'left_middle_finger', 'left_ring_finger_4', 'left_ring_finger_3', 'left_ring_finger_2',
                    'left_ring_finger', 'left_baby_finger_4', 'left_baby_finger_3', 'left_baby_finger_2',
                    'left_baby_finger']

rest_frame_names = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle',
                    'right_eye', 'right_ear', 'right_big_toe', 'right_small_toe', 'right_heel', 'right_palm',
                    'right_thumb_4', 'right_thumb_3', 'right_thumb_2', 'right_thumb', 'right_index_finger_4',
                    'right_index_finger_3', 'right_index_finger_2', 'right_index_finger', 'right_middle_finger_4',
                    'right_middle_finger_3', 'right_middle_finger_2', 'right_middle_finger', 'right_ring_finger_4',
                    'right_ring_finger_3', 'right_ring_finger_2', 'right_ring_finger', 'right_baby_finger_4',
                    'right_baby_finger_3', 'right_baby_finger_2', 'right_baby_finger', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_hip', 'left_knee', 'left_ankle',
                    'left_eye', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'left_palm',
                    'left_thumb_4', 'left_thumb_3', 'left_thumb_2', 'left_thumb', 'left_index_finger_4',
                    'left_index_finger_3', 'left_index_finger_2', 'left_index_finger', 'left_middle_finger_4',
                    'left_middle_finger_3', 'left_middle_finger_2', 'left_middle_finger', 'left_ring_finger_4',
                    'left_ring_finger_3', 'left_ring_finger_2', 'left_ring_finger', 'left_baby_finger_4',
                    'left_baby_finger_3', 'left_baby_finger_2', 'left_baby_finger', 'nose', 'neck', 'MidHip']

correct_indecis = [64, 65, 0, 1, 2, 32, 33, 34, 66, 3, 4, 5, 35, 36, 37, 6, 38, 7, 39, 40, 41, 42, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
video_keypoints = np.load(os.path.join("output", "test_with_rotation_calculation.npy"))
corrected_keypoints = []
for index, frame in enumerate(video_keypoints):
    if index==0:
        print("Dont Change ", index)
        corrected_keypoints.append(frame)
    else:
        rearranged_frame = frame[correct_indecis]
        corrected_keypoints.append(rearranged_frame)

corrected_keypoints = np.asarray(corrected_keypoints)
np.save('output\corrected_keypoints.npy', corrected_keypoints)

for wrong_keypoint in first_frame_names:
    index = rest_frame_names.index(wrong_keypoint)
    correct_indecis.append(index)

print(correct_indecis)

video_keypoints = np.load(os.path.join("output", "test_with_rotation_calculation.npy"))

print(video_keypoints[0])
print(video_keypoints[1])


output_folder = r'output\datas'
for i in range(video_keypoints.shape[0]):
    filename = os.path.join(output_folder,  '3d_data' + str(i)+'.txt')
    print(filename)
    file = open(filename, 'w')
    x = []
    y = []
    z = []
    intermidiate_data = []
    frame_data = []
    frame_keypoints = video_keypoints[i] #add spine as 68th keypoint
    spine_location = np.array(((frame_keypoints[8][0] + frame_keypoints[1][0])/2,
                              (frame_keypoints[8][1] + frame_keypoints[1][1])/2,
                              (frame_keypoints[8][2] + frame_keypoints[1][2])/2))
    all_68_keypoints = np.vstack((frame_keypoints, spine_location))
    for keypoint in all_68_keypoints:
        x.append(keypoint[0])
        y.append(480-keypoint[1])
        z.append(keypoint[2])
    intermidiate_data.append(x)
    intermidiate_data.append(y)
    intermidiate_data.append(z)
    frame_data.append(intermidiate_data)
    file.write(str(frame_data))
    file.close
