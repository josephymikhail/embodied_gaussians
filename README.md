Steps for Getting Camera Pose Estimation:
1. First you need to get the intrinsics of the camera. Place the checkerboard somewhere on the table and then take at least 10 images of it from different angles. Run calibrate.py which will use these images to get the intrinsics of your camera. When running this you can check if there any images that aren't detecting corners well and remove them from the calibration.
2. Once you have the camera intrinsics input those manually into get_pose.py. Then place the camera and checkerboard at a fixed location and run get_pose.py. 
