![alt text](https://github.com/mshakeelt/fitPartner/blob/main/test_media/LOGO.PNG)

# fitPartner
fitPartner is an experimental fitness trainer based on AI. The repository contains inference code for the pose estimation using following three SOTA keypoint detection frameworks

* [TRT Pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
* [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

For installation of these libraries please go to the respective links. JFYI Some below demos does not support the later 2 backends. I might add them later!

```
python main.py
```
After creating the development enviornment, run the above command which will load a test video and capture the video from the webcam and compare the motion of webcam with the test video in a multithreaded enviornment. It will also write the frame by frame comparison results on a text file.

```
python demo.py
```
Run the above command for a live multithreaded GUI demonstration. You will be greated with the following screen (with the non blank second screen). Left video is the person being followed and right video will be from your webcam. Please edit the line number 286 and give the path to the video file to be followed. 

![alt text](https://github.com/mshakeelt/fitPartner/blob/main/test_media/demo.png)

```
python multimodel_single_video_demo.py
```
This script let's you test all three backends.

```
python multicamera_z_coordinate_estimator.py
```
This script will estimate relative z-coordinate in a three camera setup, one camera at front, second at right, and third at left side of the participant.

```
python motion_extractor_numpy.py
```
Run the above command to write frame by frame xyz coordinates to a numpy file. It will use OpenPose backend.

# Blender Animation

animate_blender.py is the experimental script which can be used to replicate the real world motion to a blender character. It uses the xyz coordinates from the above numpy file and animate set of cubes in blender as human joints. 
