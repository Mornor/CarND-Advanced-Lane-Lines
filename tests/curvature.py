# Custom method
def plot_peaks_histogram(image):

	slices = slice_image(image)

	# Contains coordinates of the lines
	coordinates_left_line_x = []
	coordinates_left_line_y = []
	coordinates_right_line_x = []
	coordinates_right_line_y = []

	for slice in slices:
		# Get the histogram per slice (sum the pixel value, coulumn wise)
		histogram = np.sum(slice, axis=0)

		middle_point = np.int(histogram.shape[0]/2)

		# For the x-coordinates, append the max index found in the histogram 
		coordinates_left_line_x.append(np.argmax(histogram[:middle_point]))
		coordinates_right_line_x.append(np.argmax(histogram[middle_point:]) + middle_point)

		# For the y coordinates, append the center of the current slice
		y_center = np.int(slice.shape[0]/2)
		coordinates_left_line_y.append(y_center)
		coordinates_right_line_y.append(y_center)

	# Fit a second order polynomial
	left_fit = np.polyfit(coordinates_left_line_y, coordinates_left_line_x, 2)
	right_fit = np.polyfit(coordinates_right_line_y, coordinates_right_line_x, 2)
	
	# Generate x and y values for plotting
	fity = np.linspace(0, image.shape[0]-1, image.shape[0] )
	fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
	fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]
	
	plt.plot(fit_leftx, fity, color='blue')
	plt.plot(fit_rightx, fity, color='blue')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	plt.show()