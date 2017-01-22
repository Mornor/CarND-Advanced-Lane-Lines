import sys
sys.path.insert(0, './src')
import utils

PATH_CAMERA_CAL = './camera_cal/'

def test_undistort():
	calibration_images = utils.load_images(PATH_CAMERA_CAL)
	img_points, obj_points, nx, ny = utils.get_imgpoints_objpoints(calibration_images)
	undistorted_image = utils.undistort_image(calibration_images[2], obj_points, img_points, nx, ny)
	utils.plot_diff_src_undist(calibration_images[2], undistorted_image)

test_undistort()