'''
Author: Celien Nanson [cesliens@gmail.com]
This file contains the logic pipeline which will be used to process the images
received by the camera feed to draw the lines region on the road. 
'''

import utils

PATH_CAMERA_CAL = './camera_cal/'

# Calibrate the cameras based on ./camera_cal images
calibration_images = utils.load_images(PATH_CAMERA_CAL)
img_points, obj_points, nx, ny = utils.get_imgpoints_objpoints(calibration_images)

# Apply a distortion correction to images
undistorted_image = utils.undistort_image(calibration_images[4], obj_points, img_points, nx, ny)
utils.plot_diff_images(calibration_images[4], undistorted_image)