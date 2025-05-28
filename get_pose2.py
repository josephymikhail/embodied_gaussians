import numpy as np
import cv2 as cv
import glob
from pdb import set_trace as st
from PIL import Image

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 36, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
objp *= 0.0296 * 39.3701  # meters, size of each square

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob("/home/lab/embodied_gaussians/camera_pose/image_*.png")

#index = [0, 16, 32, 48]
R_target2cam = []
t_target2cam = []
# for k in range(3):
for fname in images:
    print('found an image', fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (8,6), None)
# If found, add object points, image points (after refining them)
if ret == True:
# objpoints.append(objp[index[k]:index[k+1]])
    print('corners found')
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (8,6), (-1,-1), criteria)
    imgpoints.append(corners2)

# print(corners2[0])
# print(objp[0])

# cornerstemp = 100 * np.ones_like(corners2)
# cornerstemp[..., 0] = corners2[1, :, 0].squeeze()
# cornerstemp[..., 1] = corners2[1, :, 1].squeeze()

# Draw and display the corners
cv.drawChessboardCorners(img, (8,6), corners2, ret)
# corner0 = np.array([[191.65172, 145.63283], [0, 0]])[:, None, :]
# st()
# cv.drawChessboardCorners(img, (8,6), corner0, ret)
# Image.fromarray(img[..., ::-1])
cv.imshow('img', img)
cv.imwrite(f'/home/lab/embodied_gaussians/camera_pose/image_*.png', img)
cv.waitKey(500)

cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)
R_target2cam.append(rvecs[0])
t_target2cam.append(tvecs[0])
#R_gripper2base = [np.identity(3), np.identity(3), np.identity(3)]
#t_gripper2base = [np.zeros((3,)), np.zeros((3,)), np.zeros((3,))]
# R_cam2gripper, t_cam2gripper = cv.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
mtx = np.array([[607.84, 0, 319.402],
[0, 606.518, 237.56],
[0, 0, 1]])
success, R_vec, t, inliers = cv.solvePnPRansac(objpoints[0], imgpoints[0], mtx, distCoeffs=None)
#print(t * 0.0296 * 39.3701)  # Convert to meters then inches
target2camera_cv = np.eye(4)
r = cv.Rodrigues(R_vec)[0]
target2camera_cv[:3, :3] = r
target2camera_cv[:3, 3] = t.squeeze()
camera2target_cv = np.linalg.inv(target2camera_cv)
print(camera2target_cv)