import sys
sys.path.insert(0, './src')
import utils
import cv2

PATH_CAMERA_CAL = './camera_cal/'
PATH_TEST_IMAGES = './test_images/'

def test_undistort():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'test3.jpg')
	img_points, obj_points, nx, ny = utils.get_imgpoints_objpoints()
	undistorted_image = utils.undistort_image(original_image, obj_points, img_points, nx, ny)
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
	utils.plot_diff_images(original_image, undistorted_image, False)

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

def test_warp():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'test5.jpg')
	original_image_lines = utils.combine_gradient_color(original_image)
	
	#original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	warped_image, Minv = utils.warp(original_image_lines)
	utils.plot_diff_images(original_image, warped_image, True)

def test_hls():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'straight_lines1.jpg')
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	hls_image = utils.hls_select(original_image, thresh=(90, 255))
	utils.plot_diff_images(original_image, hls_image, True)

def test_combine_gradient_color():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'test5.jpg')
	hls_image = utils.combine_gradient_color(original_image)
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	utils.plot_diff_images(original_image, hls_image, True)

def test_get_polynomials_curve():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'test5.jpg')
	original_image_lines = utils.combine_gradient_color(original_image)
	warped_image, Minv = utils.warp(original_image_lines)
	utils.get_polynomials_curve(warped_image)

def test_line_curvature():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'straight_lines1.jpg')
	original_image_lines = utils.combine_gradient_color(original_image)
	warped_image, Minv = utils.warp(original_image_lines)
	left_fit, right_fit = utils.get_polynomials_curve(warped_image)
	utils.get_line_curvature(warped_image, left_fit, right_fit)

def test_draw_lines():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'test3.jpg')
	img_points, obj_points, nx, ny = utils.get_imgpoints_objpoints()
	undistorted_image = utils.undistort_image(original_image, obj_points, img_points, nx, ny)

	original_image_lines = utils.combine_gradient_color(undistorted_image)
	warped_image , Minv = utils.warp(original_image_lines)
	left_fit, right_fit = utils.get_polynomials_curve(warped_image)
	utils.draw_lines(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB), warped_image, left_fit, right_fit, Minv)

def test_pipeline():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'straight_lines1.jpg')
	img_points, obj_points, nx, ny = utils.get_imgpoints_objpoints()
	undistorted_image = utils.undistort_image(original_image, obj_points, img_points, nx, ny)
	filtered_image = utils.combine_gradient_color(undistorted_image)
	warped_image, Minv = utils.warp(filtered_image)
	left_fit, right_fit = utils.get_polynomials_curve(warped_image)
	left_curvrad, right_curvrad = utils.get_line_curvature(warped_image, left_fit, right_fit)
	result = utils.draw_lines(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB), warped_image, left_fit, right_fit, Minv)

def test_print_data(): 
	original_image = cv2.imread(PATH_TEST_IMAGES + 'straight_lines1.jpg')
	#img_points, obj_points, nx, ny = utils.get_imgpoints_objpoints()
	#undistorted_image = utils.undistort_image(original_image, obj_points, img_points, nx, ny)
	filtered_image = utils.combine_gradient_color(original_image)
	warped_image, Minv = utils.warp(filtered_image)
	left_fit, right_fit = utils.get_polynomials_curve(warped_image)
	left_curvrad, right_curvrad = utils.get_line_curvature(warped_image, left_fit, right_fit)
	result = utils.draw_lines(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), warped_image, left_fit, right_fit, Minv)
	result = utils.draw_measured_curvature(result, left_curvrad, right_curvrad, "-0.45")
	utils.plot_image(result, False)




# test_undistort()
# test_abs_sobel_thresh()
# test_mag_thresh()
# test_dir_threshold()
# test_get_composed_tresholded_image()
# test_hls()
# test_combine_gradient_color()
# test_warp()
# test_get_polynomials_curve()
test_line_curvature()
# test_draw_lines()
# test_pipeline()
# test_print_data()