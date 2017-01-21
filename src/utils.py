'''
Author: Celien Nanson [cesliens@gmail.com]
Contains all the methods wich are used into process.py
'''

import cv2
import os
import matplotlib.image as mpimg
import numpy as np

# Define constants
PATH_CAMERA_CAL = './camera_cal/'

def calibrate_camera(): 
	images = load_camera_cal_images()

def load_camera_cal_images():
	'''
	Load the camera images and return them in an np.array ad RGB images
	'''
	images = np.array([cv2.imread(PATH_CAMERA_CAL + image) for image in os.listdir(PATH_CAMERA_CAL)])
	images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images])
	return images