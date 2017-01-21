'''
Author: Celien Nanson [cesliens@gmail.com]
This file contains the logic pipeline which will be used to process the images
received by the camera feed to draw the lines region on the road. 
'''

import utils

PATH_CAMERA_CAL = './camera_cal/'

# Calibrate the cameras based on ./camera_cal images
calibration_images = utils.load_images(PATH_CAMERA_CAL)
img_points, obj_points = utils.get_imgpoints_objpoints(calibration_images)

# Apply a distortion correction to images
undistort_image = utils.undistort_image(calibration_images[4], obj_points, img_points)
utils.plot_diff_src_undist(calibration_images[4], undistort_image)