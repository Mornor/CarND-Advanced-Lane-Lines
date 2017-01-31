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

def test_warp():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'straight_lines1.jpg')
	original_image_lines = utils.combine_gradient_color(original_image)
	original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	warped_image = utils.warp(original_image_lines)
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

def test_plot_peaks_histogram():
	original_image = cv2.imread(PATH_TEST_IMAGES + 'straight_lines1.jpg')
	original_image_lines = utils.combine_gradient_color(original_image)
	warped_image = utils.warp(original_image_lines)
	utils.plot_peaks_histogram(warped_image)


# test_undistort()
# test_abs_sobel_thresh()
# test_mag_thresh()
# test_dir_threshold()
# test_get_composed_tresholded_image()
# test_hls()
# test_combine_gradient_color()
# test_warp()
test_plot_peaks_histogram()



'''
	min_pix = 50 # Minimum number of pixel to recentre the windows
	margin=100
	left_lane_inds = []
	right_lane_inds = []

	nonzero = image.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	for slice in slices:

		histogram = np.sum(slice, axis=0)

		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint
		leftx_current = leftx_base
		rightx_current = rightx_base

		win_y_low = image.shape[0] - (slice+1)*slice.shape[0]
		win_y_high = image.shape[0] - slice*slice.shape[0]
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

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

'''