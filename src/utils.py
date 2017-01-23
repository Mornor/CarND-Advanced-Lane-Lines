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

def combine_gradient_color(image):
	'''
	Combine gradient thresholds and color space to get the best result.
	All the techniques are combined to better detect the lines
 	[TODO] : Give an undistorded image (BGR) as param
	'''
	rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Get the image with combined thresholds applied
	combined_thresholds_image = get_composed_tresholded_image(image)

	# Get hls image
	hls_image = hls_select(rgb_image, thresh=(170, 255))

	# Return the combination of both transformation
	result = np.zeros_like(hls_image)
	result[(combined_thresholds_image == 1) | (hls_image == 1)] = 1
	return result


# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def warp(image):
	# Get the image size
	image_size = (image.shape[1], image.shape[0])

	# src coordinates
	src = np.float32(
		[[850, 320],   # top right
		 [865, 450],   # bottom right
		 [533, 350],   # bottom left
		 [535, 210]])  # top left

	# dest coordinates
	dest = np.float32(
		[[870, 240],   # top right
		 [870, 370],   # bottom right
		 [520, 370],   # bottom left
		 [520, 240]])  # top left

	# Compute the perspective transform
	M = cv2.getPerspectiveTransform(src, dst)

	# Create waped image
	#warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
	warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)  # keep same size as input image

	return warped


def get_composed_tresholded_image(image):
	ksize = 3 # Choose a larger odd number to smooth gradient measurements

	# Apply each of the thresholding functions
	gradx = abs_sobel_thresh(image, orientation='x', sobel_kernel=ksize, thresh=(20, 100))
	grady = abs_sobel_thresh(image, orientation='y', sobel_kernel=ksize, thresh=(20, 100))
	mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=(35, 100))
	dir_binary = dir_thresh(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

	combined = np.zeros_like(dir_binary)
	combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
	
	return combined
	
def abs_sobel_thresh(image, orientation, sobel_kernel=3, thresh=(0, 255)):
	'''
	Apply SobelX (by default), take the absolute value and apply a threshold to create a binary mask 
	@return Image with SobelX or SobelY with the defined mask applied
	''' 
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Calcul derivative in the x and y direction
	sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	# Absolute value of the x derivatives, converted to 8 bits
	if(orientation == 'x'):
		abs_sobel = np.absolute(sobelx)
	else: 
		abs_sobel = np.absolute(sobely)
	
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

	# Create a binary threshold to select pixels based on gradient strength (here on x direction):
	# Pixel corresponding to the condition provided by thresh_min and thresh_max will be 1. The others, 0. 
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	return binary_output

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
	'''
	Apply a threshold to the overall magnitude of the gradient, in x and y directions.
	'''
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Calcul derivative in the x and y direction
	sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	# Calculate the gradient magnitude, and rescale to 8 bits
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	scale_factor = np.max(gradmag)/255 
	gradmag = (gradmag/scale_factor).astype(np.uint8) 

	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

	# Return the binary image
	return binary_output

def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
	'''
	Detect vertical lines. 
	Compute the direction of the gradient (arctan2(np.absolute(sobely), np.absolute(sobelx))) and apply it to each pixel of the image.
	Each pixel will contains a value for the angle of the gradient away from horizontal in units of radians. 
	A value of (+/-) pi/2 indicate a vertical line.
	Apply this to the image as a mask, and return it.
	'''
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	
	# Take the absolute value of the gradient direction, 
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
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
	# img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
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
	plt.plot(570, 320, '.')
	plt.show()

def plot_diff_images(original_image, undistorted_image, gray):
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
	f.tight_layout()
	ax1.imshow(original_image)
	ax1.set_title('Original image', fontsize=25)
	if(gray):
		ax2.imshow(undistorted_image, cmap='gray')
	else: 
		ax2.imshow(undistorted_image)
	ax2.set_title('Image with combined thresholds\n + Accentuate S space in HLS', fontsize=25)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()