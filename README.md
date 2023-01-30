# fitPartner
fitPartner is an experimental fitness trainer based on AI. The repository contains inference code for the pose estimation using following three SOTA keypoint detection frameworks

* [TRT Pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
* [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

(JFYI Some demos does not support the later 2 backends. I might add them later!)

```
$ python main.py
```
After creating the development enviornment, run the above command which will load a test video and capture the video from the webcam and compare the motion of webcam with the test video in a multithreaded enviornment. It will also write the frame by frame comparison results on a text file.

```
$ python demo.py
```
Run the above command for a live multithreaded GUI demonstration.

```
$ python multimodel_single_video_demo.py
```
This script let's you test all three backends.

```
$ python multicamera_z_coordinate_estimator.py
```
This script will estimate relative z-coordinate in a three camera setup, one camera at front, second at right, and third at left side of the participant.

```
$ python motion_extractor_numpy.py
```
Run the above command to write frame by frame xyz coordinates to a numpy file. It will use OpenPose backend.
