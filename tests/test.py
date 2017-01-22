import sys
sys.path.insert(0, './src')
import utils
import cv2

PATH_CAMERA_CAL = './camera_cal/'
PATH_TEST_IMAGES = './test_images/'

def test_undistort():
	calibration_images = utils.load_images(PATH_CAMERA_CAL)
	img_points, obj_points, nx, ny = utils.get_imgpoints_objpoints(calibration_images)
	undistorted_image = utils.undistort_image(calibration_images[2], obj_points, img_points, nx, ny)
	utils.plot_diff_images(calibration_images[2], undistorted_image)

def test_thresholded():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'test3.jpg')
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	thresholded_image = utils.get_tresholded_image(original_image, 20, 100)
	utils.plot_diff_images(original_image, thresholded_image, True)

# test_undistort()
test_thresholded()