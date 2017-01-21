'''
Author: Celien Nanson [cesliens@gmail.com]
Contains all the methods wich are used into process.py
'''

import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Define constants
PATH_CAMERA_CAL = './camera_cal/'

def get_imgpoints_objpoints(calibration_images): 
	'''
	Compute the camera calibration matrix and distortion coefficients.
	'''
	nx = 9 # nb corner along x axis
	ny = 6 # nb corner along y axis

	for image in calibration_images: 
		# 3D points of the received imaes (from real world). Know object coordinates from the chessboard
		#  X = (0 -> 7) and Y = (0 -> 5) and Z (always 0 since it is a plane)
		obj_points = []
		img_points = [] # 2D points im image plane

		# Create a 6*8 points np array (1, 2, 0) or (2, 4, 0)...
		objp = np.zeros([ny*nx, 3], np.float32)
		objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) # Set x and y coordinates

		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Find the chessboard corner
		ret, corners = cv2.findChessboardCorners(gray_image, (nx, ny), None)

		print(ret)
		if(ret):
			img_points.append(corners)
			obj_points.append(objp)

	return img_points, obj_points

def undistort_image(img, objpoints, imgpoints):
	'''
	Takes an image, object points, and image points
	Performs the camera calibration, image distortion correction and 
	returns the undistorted image
	'''
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
	img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	return dst

def load_images(dir_path):
	'''
	Load the camera images and return them in an np.array ad RGB images
	@return np array of images within the @dir_path
	'''
	return np.array([cv2.imread(dir_path + image) for image in os.listdir(dir_path)])

def plot_image(image): 
	'''
	Display a single image
	'''
	plt.imshow(image)
	plt.show()