import cv2
import numpy as np

# === USER INPUTS ===
image_path = "/home/lab/embodied_gaussians/camera_pose/image_00.png"  # Path to your checkerboard image
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

# === LOAD IMAGE AND FIND CORNERS ===
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

if not ret:
    print("Checkerboard not detected.")
    exit()

# Refine corner accuracy
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# Draw detected corners
cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
cv2.imshow("Detected Corners", img)



# === POSE ESTIMATION ===
success, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

# Convert rvec to rotation matrix
R, _ = cv2.Rodrigues(rvec)

# === REPROJECTION ERROR ===
imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
reprojection_error = cv2.norm(corners2, imgpoints2, cv2.NORM_L2) / len(imgpoints2)

'''
axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1]])  # X (red), Y (green), Z (blue)
imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
corner = tuple(corners2[0].ravel().astype(int))
img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 5)  # X
img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)  # Y
img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 5)  # Z

cv2.imshow('Pose Axes', img)
cv2.waitKey(0)
'''



cv2.destroyAllWindows()

# === PRINT RESULTS ===
print("\n=== POSE ESTIMATION RESULTS ===")
print("Rotation Matrix (R):\n", R)
print("Translation Vector (inches):\n", tvec * 39.3701)
print("Reprojection Error (pixels): {:.4f}".format(reprojection_error))




#standing behind the camera: pos x axis is to the right, pos y axis is down, pos z axis is forward
