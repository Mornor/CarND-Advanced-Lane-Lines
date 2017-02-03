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
	filtered_image = utils.combine_gradient_color(undistorted_image)

	# Perspective transform, warp the image to better detect line curvature (Bird Eye view)
	warped_image, Minv = utils.warp(filtered_image)

	# Get the polynomials fitting the curvature of the lines
	left_fit, right_fit = utils.get_polynomials_curve(warped_image)

	# Measure the curvature of the two lines
	left_curvrad, right_curvrad = utils.get_line_curvature(warped_image, left_fit, right_fit)

	# Draw back the lines on the input image
	result = utils.draw_lines(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB), warped_image, left_fit, right_fit, Minv)

	# Retunn the result
	return result


output = 'output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)