import cv2
import numpy as np
import pyrealsense2 as rs

# === USER INPUTS ===
CHECKERBOARD = (8, 6)  # (columns, rows) of inner corners
square_size = 0.036  # size of one square in meters

# Intrinsic parameters (replace with your values)
#from real sense
#fx = 607.841552734375
#fy = 606.5180053710938
#cx = 319.40252685546875
#cy = 237.56298828125

#from manual calibration with 20 images
fx = 538.3586
fy = 546.76
cx = 281.4157
cy = 277.938

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])
dist_coeffs = np.array([0.2514, -0.78757, 0.01096, -0.02877, 0.43484])
#dist_coeffs = np.array([0,0,0,0])

# === PREPARE OBJECT POINTS ===
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

pipeline = rs.pipeline()
config = rs.config()
config.enable_device('007522062003')
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

cv2.namedWindow("RealSense Checkerboard", cv2.WINDOW_AUTOSIZE)
output_freq = 0

try:
    while True:
        output_freq += 1
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:


            # Refine the corner locations
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
            )

            # Solve PnP to get camera pose
            success, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

            if success:
                # Convert to a 4x4 homogeneous transformation matrix
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R
                # Convert to inches for better intuition
                #T[:3, 3] = tvec.ravel() * 39.3701
                T[:3, 3] = tvec.ravel()
                inv_T = np.linalg.inv(T)


                if output_freq % 150 == 0:
                    print(T)

                # Draw checkerboard corners on the image
                cv2.drawChessboardCorners(color_image, CHECKERBOARD, corners2, ret)

                # === REPROJECTION ERROR ===
                imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
                reprojection_error = cv2.norm(corners2, imgpoints2, cv2.NORM_L2) / len(imgpoints2)


        cv2.imshow("RealSense Checkerboard", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

#standing behind the camera: pos x axis is to the right, pos y axis is down, pos z axis is forward
