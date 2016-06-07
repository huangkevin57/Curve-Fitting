import cv2
import numpy as np
import sys

# color is the specific road marking color output by SegNet
color = np.array([0,69,255])

# Other colors. Note: These may be different based on your computer
green = np.array([0,128,0])
white = np.array([255,255,255])
gray = np.array([128,128,128])
yellow = np.array([255,255,0])
red = np.array([255,0,0])
cyan = np.array([0,255,255])
maroon = np.array([128,0,0])
black = np.array([0,0,0])
blue = np.array([255,0,0])
magenta = np.array([255,0,255])

if len(sys.argv) != 2:
	print "Usage: ./curvefit.py image.png"
	sys.exit()
inputFilename = sys.argv[1]
outputFilename = "Output.png"
src = cv2.imread(inputFilename, 1)
cv2.destroyAllWindows()
# Create binary image indicating if pixel was categorized as "road marking"
binary = cv2.inRange(src, color, color)
screen_height, screen_width = binary.shape

point_size = 7
tan = .12
# Base number of pixels to determine whether two pixels are close enough to be classified
# under the same line. This number decreases the higher the pixels are on the screen, to
# compensate for higher pixels being further apart than they appear (in terms of their
# actual distance on the road)
connected_base = 15
# The connected_base, but will be made smaller as described above
connected_thresh = 15 #say 5% of screen_height

# Take out non-road markings from res (which is the return image)
res = cv2.bitwise_and(src,src, mask = binary)


# Draw a green point of size point_size at coordinates (x, y)
def point(img, x, y):
	h_min = max(x - point_size / 2, 0)
	h_max = min(x + 1 + point_size / 2, screen_height)
	w_min = max(y - point_size / 2, 0)
	w_max = min(y + 1 + point_size / 2, screen_width)
	for i in range(h_min, h_max):
		for j in range(w_min, w_max):
			img[i, j] = green

# Also draws a point, but of smaller size and specified color
def line(img, x, y, color):
	h_min = max(x - 1, 0)
	h_max = min(x + 2, screen_height)
	w_min = max(int(y - 1), 0)
	w_max = min(int(y + 2), screen_width)
	for i in range(h_min, h_max):
		for j in range(w_min, w_max):
			img[i, j] = color

# Evaluates polynomial at point x, given the polynomial coefficients array
def evaluation(poly, x):
	ret = 0
	for i in range(len(poly)):
		ret += poly[i] * (x ** (len(poly) - i - 1))
	return ret

# For debugging: Gives a sense of what 15 frames looks like
# for i in range(15):
# 	res[100, i] = green

# Denotes whether a pixel has been visited. Becomes line number once the pixel is assigned to
# a line.
visited = [[-1 for i in range(screen_width)] for j in range(screen_height)]

# Number of lines produced in image
line_count = 0
# Stores an array of coordinates for each line
lines = []


# Create an array of size screen_width that starts from the center and alternates left
# and right (e.g. if screen_width = 7, then outputs [3,4,2,5,1,6,0])
w_from_center = [-1 for i in range(screen_width)]
w_center = (screen_width - 1) / 2
w_from_center[0] = w_center
w_from_center[1] = w_center + 1
for i in range(3, screen_width, 2):
	w_from_center[i] = w_from_center[i - 2] + 1
for i in range(2, screen_width, 2):
	w_from_center[i] = w_from_center[i - 2] - 1


# Go down the screen and for every road marking pixel, categorize it as the same line as nearby pixels
# If no nearby categorized pixels already exist, categorize this pixel under a new line
# Visited indicates what line a pixel has been classified as (starts out at -1, becomes 0 if not road
# marking, otherwise becomes positive integer based on line categorized under)
for h in range(screen_height):
	# Make the connected threshold smaller as we are higher on the screen, since physical distances
	# are larger than they appear
	connected_thresh = max(1, int((h - .5 * screen_height) / (.5 * screen_height) * connected_base), 1)
	for w in w_from_center:
		# If road marking
		if binary[h, w]:		
			# Look at neighbors starting from adjacent pixels, then go outwards until either
			# finding a categorized pixel or meeting the connected threshold
			for i in range(1, connected_thresh + 1):
				# Look at vertical strips
				if visited[h][w] == -1:
					for j in range(max(0, h - i), min(screen_height, h + i + 1)):
						if w + 2 * i - 1 < screen_width and visited[j][w + 2 * i - 1] > 0:
							lines[visited[j][w + 2 * i - 1] - 1].append((h,w))
							visited[h][w] = visited[j][w + 2 * i - 1]
							break
						if w + 2 * i < screen_width and visited[j][w + 2 * i] > 0:
							lines[visited[j][w + 2 * i] - 1].append((h,w))
							visited[h][w] = visited[j][w + 2 * i]
							break
						if w - 2 * i + 1 >= 0 and visited[j][w - 2 * i + 1] > 0:
							lines[visited[j][w - 2 * i + 1] - 1].append((h,w))
							visited[h][w] = visited[j][w - 2 * i + 1]
							break
						if w - 2 * i >= 0 and visited[j][w - 2 * i] > 0:
							lines[visited[j][w - 2 * i] - 1].append((h,w))
							visited[h][w] = visited[j][w - 2 * i]
							break
				# Look at horizontal strips (Note: these are twice as long as vertical strips. this is
				# to compensate for the road lines going horizontally into the center of the screen as
				# their height increases)
				if visited[h][w] == -1:
					for j in range(max(0, w - 2 * i), min(screen_width, w + 2 * i + 1)):
						if h + i < screen_height and visited[h + i][j] > 0:
							lines[visited[h + i][j] - 1].append((h,w))
							visited[h][w] = visited[h + i][j]
							break
						if h >= i and visited[h - i][j] > 0:
							lines[visited[h - i][j] - 1].append((h,w))
							visited[h][w] = visited[h - i][j]
							break
				else:
					break
			# If the pixel is still uncategorized after looking at all close by pixels, categorize
			# it under a new line and increment the total line count
			if visited[h][w] == -1:
				lines.append([(h,w)])
				line_count += 1
				visited[h][w] = line_count
		# not road marking
		else:
			visited[h][w] = 0

# Find irrelevant lines and color them cyan, otherwise fit relevant lines and draw tangent
invalid_lines = []
for i in range(len(lines)):
	minimum = screen_height - 1
	maximum = 0
	for (h,w) in lines[i]:
		if h < minimum:
			minimum = h
		if h > maximum:
			maximum = h
	diff = maximum - minimum
	# If length of line is too small, or line doesn't start low enough, or line isnt large enough,
	# classify it as irrelevant
	if diff < .15 * screen_height or maximum < screen_height / 2.0 or len(lines[i]) < 100:
		invalid_lines.append(i)
		for (h,w) in lines[i]:
			res[h,w] = cyan
	# Otherwise, the line is relevant. Divide the line into sections (larger sections at the
	# lower end compared to the higher end, to compensate for higher lengths being larger
	# than they appear) 
	else:
		array = np.array([1., 3.5, 3.,2.5, 2.,1.5])
		n_array = array / sum(array)
		beginning = maximum
		h_array = [maximum for k in range(len(array) + 1)]
		for j in range(len(array)):
			h_array[j + 1] = h_array[j] - n_array[j] * diff
		section = [[] for j in range(len(array))]
		for (h,w) in lines[i]:
			for j in range(len(array)):
				if h >= h_array[j + 1]:
					section[j].append((h,w))
					break
		points_to_fit_h = []
		points_to_fit_w = []
		# Average the point coordinates in each section, to produce a point that is "representative"
		# of this entire section, to be used for the polynomial fit.
		for j in range(len(section)):
			h_avg = 0
			w_avg = 0
			count = 0
			for (h,w) in section[j]:
				h_avg += h
				w_avg += w
				count += 1
			if count == 0:
				continue
			else:
				# If there are too many points in this section (say, twice as many as there
				# there should be proportionally in this section, then we're probably capturing
				# a stop line within the side line, so all points in this section out)
				if count / n_array[j] < 2 * len(lines[i]):
					h_avg /= count
					w_avg /= count
					point(res, h_avg, w_avg)
					points_to_fit_h.append(h_avg)
					points_to_fit_w.append(w_avg)
				else:
					new_line = []
					for (h,w) in lines[i]:
						if h < h_array[j + 1] or h >= h_array[j]:
							new_line.append((h,w))
						else:
							res[h,w] = cyan
					lines[i] = new_line

		# Polynomial fit representative points
		p = np.polyfit(points_to_fit_h, points_to_fit_w, 3)
		for i in range(minimum, maximum):
			line(res, i, evaluation(p, i), white)
		# Extend line, based off first tan (~12%) of polynomial fit
		h0 = points_to_fit_h[0]
		w0 = points_to_fit_w[0]
		h1 = max(0, (int) (points_to_fit_h[0] - tan * screen_height))
		w1 = (int) (evaluation(p, h1))
		slope = 1.0 * (w0 - w1) / (h0 - h1)
		intercept = 1.0 * (w1 * h0 - h1 * w0) / (h0 - h1)
		for k in range(0, h0 + 1):
			line(res, k, slope * k + intercept, blue)

### For debugging purposes; separates pixels into their line categories. Note: for different
### systems, these colors might not be accurate.
# for i in range(screen_height):
# 	for j in range(screen_width):
# 		if visited[i][j] == 1:
# 			res[i,j] = green
# 		elif visited[i][j] == 2:
# 			res[i,j] = white
# 		elif visited[i][j] == 3:
# 			res[i,j] = gray
# 		elif visited[i][j] == 4:
# 			res[i,j] = magenta
# 		elif visited[i][j] == 5:
# 			res[i,j] = red
# 		elif visited[i][j] == 6:
# 			res[i,j] = cyan
# 		elif visited[i][j] == 7:
# 			res[i,j] = maroon
# 		elif visited[i][j] == 8:
# 			res[i,j] = green
# 		elif visited[i][j] == 9:
# 			res[i,j] = white
# 		elif visited[i][j] == 10:
# 			res[i,j] = gray
# 		elif visited[i][j] == 11:
# 			res[i,j] = magenta
# 		elif visited[i][j] == 12:
# 			res[i,j] = red
# 		elif visited[i][j] == 13:
# 			res[i,j] = cyan
# 		elif visited[i][j] == 14:
# 			res[i,j] = maroon

cv2.imshow('Road Fit',res)
cv2.waitKey(0)
cv2.imwrite(outputFilename, res)
