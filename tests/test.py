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
	utils.plot_diff_images(calibration_images[2], undistorted_image, False)

def test_abs_sobel_thresh():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'test3.jpg')
	thresholded_image = utils.abs_sobel_thresh(original_image, 'y', sobel_kernel=3, thresh=(20, 100))
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	utils.plot_diff_images(original_image, thresholded_image, True)

def test_mag_thresh(): 
	original_image = cv2.imread(PATH_TEST_IMAGES + 'test3.jpg')
	thresholded_image = utils.mag_thresh(original_image, sobel_kernel=3, thresh=(40, 100))
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	utils.plot_diff_images(original_image, thresholded_image, True)

def test_dir_threshold(): 
	original_image = cv2.imread(PATH_TEST_IMAGES + 'test3.jpg')
	thresholded_image = utils.dir_thresh(original_image, sobel_kernel=15, thresh=(0.7, 1.3))
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	utils.plot_diff_images(original_image, thresholded_image, True)

def test_get_composed_tresholded_image():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'udacity_test.png')
	thresholded_image = utils.get_composed_tresholded_image(original_image)
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	utils.plot_diff_images(original_image, thresholded_image, True)


# test_undistort()
# test_abs_sobel_thresh()
# test_mag_thresh()
# test_dir_threshold()
# test_get_composed_tresholded_image()