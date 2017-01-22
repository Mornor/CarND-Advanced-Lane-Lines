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

def get_tresholded_image(image, orientation, thresh_min, thresh_max):
	'''
	Apply mask to get threshodled image (Sobel Y and Gradient threshold)
	@return Thresholded image
	''' 
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Calcul derivative in the x and y direction
	sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
	sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)

	# Absolute value of the x derivatives, converted to 8 bits
	if(orientation == 'x'):
		abs_sobel = np.absolute(sobelx)
	else: 
		abs_sobel = np.absolute(sobely)
	
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

	# Create a binary threshold to select pixels based on gradient strength (here on x direction):
	# Pixel corresponding to the condition provided by thresh_min and thresh_max will be 1. The others, 0. 
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	return binary_output


def get_imgpoints_objpoints(calibration_images): 
	'''
	Compute the camera calibration matrix and distortion coefficients.
	@return imgpoint and objpoints
	'''
	nx = 9 # nb corner along x axis
	ny = 6 # nb corner along y axis

	# 3D points of the received imaes (from real world). Know object coordinates from the chessboard
	#  X = (0 -> 7) and Y = (0 -> 5) and Z (always 0 since it is a plane)
	obj_points = []
	img_points = [] # 2D points im image plane

	# Create a 6*8 points np array (1, 2, 0) or (2, 4, 0)...
	objp = np.zeros([ny*nx, 3], np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) # Set x and y coordinates

	for image in calibration_images: 
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Find the chessboard corner
		ret, corners = cv2.findChessboardCorners(gray_image, (nx, ny), None)

		print(ret)
		if(ret):
			img_points.append(corners)
			obj_points.append(objp)

	return img_points, obj_points, nx, ny

def undistort_image(img, objpoints, imgpoints, nx, ny):	
	'''
	Takes an image, object points, and image points
	Performs the camera calibration and image distortion correction
	@return the undistorted image
	'''
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
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
	plt.imshow(image)
	plt.show()

def plot_diff_images(original_image, undistorted_image, gray):
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
	f.tight_layout()
	ax1.imshow(original_image)
	ax1.set_title('Original Image', fontsize=30)
	if(gray):
		ax2.imshow(undistorted_image, cmap='gray')
	else: 
		ax2.imshow(undistorted_image)
	ax2.set_title('Image with Sobel Y', fontsize=30)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()