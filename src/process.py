'''
Author: Celien Nanson [cesliens@gmail.com]
This file contains the logic pipeline which will be used to process the images
received by the camera feed to draw the lines region on the road. 
'''

import utils
import cv2
from moviepy.editor import VideoFileClip

PATH_CAMERA_CAL = './camera_cal/'

# Calibrate the cameras based on ./camera_cal images
img_points, obj_points, nx, ny = utils.get_imgpoints_objpoints()

def process_image(image):

	# Undistort image 
	undistorted_image = utils.undistort_image(image, obj_points, img_points, nx, ny)

	# Apply a combination of different filter thresholds and color space changes to the image to make line easy to detect
	#filtered_image = utils.combine_gradient_color(undistorted_image)
	filtered_image = utils.get_composed_tresholded_image(undistorted_image)

	# Perspective transform, warp the image to better detect line curvature (Bird Eye view)
	warped_image, Minv = utils.warp(filtered_image)

	# Get the polynomials fitting the curvature of the lines
	left_fit, right_fit = utils.get_polynomials_curve(warped_image)

	# Measure the curvature of the two lines, and get the distance from the center
	left_curvrad, right_curvrad, dst_from_center = utils.get_line_curvature(warped_image, left_fit, right_fit)

	# Draw the detected lines on the input image
	result = utils.draw_lines(undistorted_image, warped_image, left_fit, right_fit, Minv)

	# Print the computed curvature on the input image, and the distance from the center
	result = utils.draw_measured_curvature(result, left_curvrad, right_curvrad, dst_from_center)

	# Return the result
	return result


output = 'output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)