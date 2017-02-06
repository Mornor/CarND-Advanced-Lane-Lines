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

def draw_lines(original_image, warped_image, left_fit, right_fit, Minv):

	ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped_image).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (warped_image.shape[1], warped_image.shape[0])) 
	
	# Combine the result with the original image
	result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
	#plt.imshow(result)
	#plt.show()
	return result

def get_line_curvature(image, left_fit, right_fit):

	out_img = np.dstack((image, image, image))*255

	ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

	'''
	mark_size = 3
	plt.xlim(0, 1280)
	plt.ylim(0, 720)
	plt.plot(left_fitx, ploty, color='green', linewidth=3)
	plt.plot(right_fitx, ploty, color='green', linewidth=3)
	plt.gca().invert_yaxis() # to visualize as we do the images
	plt.show()
	'''

	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
	
	# Calculate the new radius of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	
	# Now our radius of curvature is in meters
	return left_curverad, right_curverad

def get_polynomials_curve(image):

	# Metters per pixel
	xm_per_pix = 3.7/700 

	histogram = np.sum(image[image.shape[0]/2:,:], axis=0)
	out_img = np.dstack((image, image, image))*255

	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = image.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	leftx_current = leftx_base
	rightx_current = rightx_base

	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Slice the image in 10 horizonally
	slices = slice_image(image)
	window_height = np.int(slices[0].shape[0])

	for i in range(0, len(slices)):
		win_y_low = image.shape[0] - (i+1)*window_height
		win_y_high = image.shape[0] - i*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Draw the windows on the visualization image (only useful for testing)
		# cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		# cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each

	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Compute the deviation of the camera from the center
	#print(leftx.shape)
	#print(leftx[1])
	#print(midpoint)
	car_middle_pixel = int((leftx[0] + rightx[0])/2)
	screen_off_center = midpoint - car_middle_pixel
	meters_off_center = xm_per_pix * rightx[0]
	# print(meters_off_center)

	'''
	# Plot the out_img
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	plt.show()
	'''

	return left_fit, right_fit

def slice_image(image, slices=10):
	'''
	Return an array of horizontal slices of the image 
	'''
	original_height = image.shape[0]
	slices_array = []
	
	for i in range(0, slices): 
		slices_array.append(image[image.shape[0] - (original_height / float(slices)):]) # Take the bottom slide
		image = image = image[:-slices_array[i].shape[0]] # From the original image, remove this slice. 
	
	# Convert list to np array (10, 72, 1280)
	return np.asarray(slices_array)


def combine_gradient_color(image):
	'''
	Combine gradient thresholds and color space to get the best result.
	All the techniques are combined to better detect the lines
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
def hls_select(image, thresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def warp(image):
	# Get the image size
	image_size = (image.shape[1], image.shape[0])

	top_right_src = [740, 460]
	top_left_src = [575, 460]
	bottom_right_src = [1200, 720]
	bottom_left_src = [180, 720]

	vertices = np.array([[bottom_left_src, top_left_src, top_right_src, bottom_right_src]], dtype=np.int32)
	region_of_interest = extract_region_of_interest(image, vertices)

	# src coordinates
	src = np.float32([
		 top_right_src,   
		 bottom_right_src, 
		 bottom_left_src,  
		 top_left_src	
	])  	

	top_right_dest = [960, 0]
	top_left_dest = [320, 0]
	bottom_right_dest = [960, 720]
	bottom_left_dest = [320, 720]

	# dst coordinates
	dst = np.float32([
		 top_right_dest,   	
		 bottom_right_dest, 
		 bottom_left_dest,  
		 top_left_dest  	
	])

	# Compute the perspective transform
	M = cv2.getPerspectiveTransform(src, dst)

	# Compute the inverse matrix (will be used in the last steps)
	Minv = cv2.getPerspectiveTransform(dst, src)

	#pythpn3  Create waped image
	warped = cv2.warpPerspective(region_of_interest, M, image_size, flags=cv2.INTER_LINEAR)  # keep same size as input image

	#plot_diff_images(region_of_interest, warped, False)

	return warped, Minv
	

def extract_region_of_interest(image, vertices):
	"""
	Applies an image mask.

	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	# Defining a blank mask to start with
	mask = np.zeros_like(image)

	# Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(image.shape) > 2:
		channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	# Filling pixels inside the polygon defined by "vertices" with the fill color
	cv2.fillPoly(mask, vertices, ignore_mask_color)

	# Feturning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(image, mask)
	
	return masked_image

def draw_measured_curvature(image, left_curverad, right_curverad):

	# Print left radius on the left side of the image
	cv2.putText(image, 'Left radius', (50, 600), fontFace = 5, fontScale = 1.5, color=(255,255,255), thickness = 2)
	cv2.putText(image, '{}m'.format(int(left_curverad)), (70, 650), fontFace = 5, fontScale = 1.5, color=(255,255,255), thickness = 2)

	# Print left radius on the left side of the image
	cv2.putText(image, 'Right radius', (1000, 600), fontFace = 5, fontScale = 1.5, color=(255,255,255), thickness = 2)
	cv2.putText(image, '{}m'.format(int(left_curverad)), (1070, 650), fontFace = 5, fontScale = 1.5, color=(255,255,255), thickness = 2)

	# Print distance from center
	# cv2.putText(image, 'Distance from center', (370, 100), fontFace = 5, fontScale = 2, color=(255,255,255), thickness = 2)
	# cv2.putText(image, '{}m'.format(dst_from_center), (550, 160), fontFace = 5, fontScale = 2, color=(255,255,255), thickness = 2)

	return image

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

def get_imgpoints_objpoints(): 
	'''
	Compute the camera calibration matrix and distortion coefficients.
	@return imgpoint and objpoints
	'''

	calibration_images = load_images(PATH_CAMERA_CAL)

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
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

def plot_image(image, gray): 
	if(gray):
		plt.imshow(image, cmap='gray')
	else: 
		plt.imshow(image)
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
	ax2.set_title('Region of interest - warped and with filter', fontsize=25)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()