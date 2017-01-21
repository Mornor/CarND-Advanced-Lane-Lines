'''
Author: Celien Nanson [cesliens@gmail.com]
This file contains the logic pipeline which will be used to process the images
received by the camera feed to draw the lines region on the road. 
'''

import utils

PATH_CAMERA_CAL = './camera_cal/'

# Calibrate the cameras based on ./camera_cal images
calibration_images = utils.load_images(PATH_CAMERA_CAL)
print(len(calibration_images))
img_points, obj_points = utils.get_imgpoints_objpoints(calibration_images)
undistort_image = utils.undistort_image(calibration_images[2], obj_points, img_points)
utils.plot_image(calibration_images[2])
utils.plot_image(undistort_image)