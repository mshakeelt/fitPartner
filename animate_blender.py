import bpy
import numpy as np
import time

collection = bpy.data.collections.get('Collection')
for obj in collection.objects:
    bpy.data.objects.remove(obj, do_unlink=True)

ordered_keypoint_names = ['Nose', 'Neck', 'RightShoulder', 'RightElbow', 'RightWrist', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
  'Hips', 'RightHip', 'RightKnee', 'RightAnkle', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'RightEye', 'LeftEye',
  'RightEarEndSite', 'LeftEarEndSite', 'LeftBigToeEndSite', 'LeftSmallToeEndSite', 'LeftHeelEndSite',
  'RightBigToeEndSite', 'RightSmallToeEndSite', 'RightHeelEndSite', 'RightPalm', 'RightThumb1', 'RightThumb2',
  'RightThumb3', 'RightThumb4EndSite', 'RightIndex1', 'RightIndex2', 'RightIndex3', 'RightIndex4EndSite',
  'RightMiddle1', 'RightMiddle2', 'RightMiddle3', 'RightMiddle4EndSite', 'RightRing1', 'RightRing2',
  'RightRing3', 'RightRing4EndSite', 'RightBaby1', 'RightBaby2', 'RightBaby3', 'RightBaby4EndSite', 'LeftPalm',
  'LeftThumb1', 'LeftThumb2', 'LeftThumb3', 'LeftThumb4EndSite', 'LeftIndex1', 'LeftIndex2', 'LeftIndex3',
  'LeftIndex4EndSite', 'LeftMiddle1', 'LeftMiddle2', 'LeftMiddle3', 'LeftMiddle4EndSite', 'LeftRing1',
  'LeftRing2', 'LeftRing3', 'LeftRing4EndSite', 'LeftBaby1', 'LeftBaby2', 'LeftBaby3', 'LeftBaby4EndSite']



numpy_animation_file = np.load(r'output\animation.npy')
first_frame = numpy_animation_file[0]

cubes_as_keypoints = []
for index, keypoint in enumerate(first_frame):
    cube = bpy.ops.mesh.primitive_cube_add(location=(keypoint[0], 480-keypoint[1], keypoint[2]))
    ob = bpy.context.object
    ob.name = ordered_keypoint_names[index]
    cubes_as_keypoints.append(ob)

global frame
frame = 1
def every_2_seconds():
    global frame
    for index, keypoint in enumerate(numpy_animation_file[frame]):
         bpy.data.objects[ordered_keypoint_names[index]].location = (keypoint[0], 480-keypoint[1], keypoint[2])
    frame+=1
    return 0.2

bpy.app.timers.register(every_2_seconds)