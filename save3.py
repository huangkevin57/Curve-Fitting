import cv2
import numpy as np

point_size = 7

def point(img, x, y):
	h_min = max(x - point_size / 2, 0)
	h_max = min(x + 1 + point_size / 2, screen_height)
	w_min = max(y - point_size / 2, 0)
	w_max = min(y + 1 + point_size / 2, screen_width)
	for i in range(h_min, h_max):
		for j in range(w_min, w_max):
			img[i, j] = green

def line(img, x, y):
	h_min = max(x - 1, 0)
	h_max = min(x + 2, screen_height)
	w_min = max(int(y - 1), 0)
	w_max = min(int(y + 2), screen_width)
	for i in range(h_min, h_max):
		for j in range(w_min, w_max):
			#print h_min, h_max, w_min, w_max
			img[i, j] = white

def evaluation(poly, x):
	ret = 0
	for i in range(len(poly)):
		ret += poly[i] * (x ** (len(poly) - i - 1))
	return ret

inputFilename = "test4seg.png"
outputFilename = "004360-SegNet-binary.png"
src = cv2.imread(inputFilename, 1)
cv2.imwrite('messigray.png',src)

cv2.destroyAllWindows()

color = np.array([0,69,255])

green = np.array([0,128,0])
white = np.array([255,255,255])
gray = np.array([128,128,128])
yellow = np.array([255,255,0])
red = np.array([255,0,0])
cyan = np.array([0,255,255])
maroon = np.array([128,0,0])
black = np.array([0,0,0])

out = cv2.inRange(src, color, color)

screen_height, screen_width = out.shape
connected_base = 20
connected_thresh = screen_height / 24 #say 5% of screen_height

res = cv2.bitwise_and(src,src, mask= out)

for i in range(15):
	res[100, i] = green

visited = [[-1 for i in range(screen_width)] for j in range(screen_height)]

line_count = 0

lines = []

for h in range(screen_height):
	connected_thresh = max(1, int((h - .5 * screen_height) / (.5 * screen_height) * connected_base), 1)
	for w in range(screen_width):
		if out[h, w]:
			
			for i in range(1, connected_thresh + 1):
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
			if visited[h][w] == -1:
				lines.append([(h,w)])
				line_count += 1
				visited[h][w] = line_count
		else:
			visited[h][w] = -1

# for i in range(screen_height):
# 	for j in range(screen_width):
# 		if visited[i][j] == 1:
# 			res[i,j] = green
# 		elif visited[i][j] == 2:
# 			res[i,j] = white
# 		elif visited[i][j] == 3:
# 			res[i,j] = gray
# 		elif visited[i][j] == 4:
# 			res[i,j] = np.array([255,0,255])
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
# 			res[i,j] = np.array([255,0,255])
# 		elif visited[i][j] == 12:
# 			res[i,j] = red
# 		elif visited[i][j] == 13:
# 			res[i,j] = cyan
# 		elif visited[i][j] == 14:
# 			res[i,j] = maroon

#### FIND MAX - MIN Y, FIND # OF PIXELS
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
	if diff < .15 * screen_height: # or maximum < 2 * screen_height / 3.0 or len(lines[i]) < 100:
		invalid_lines.append(i)
		for (h,w) in lines[i]:
			res[h,w] = cyan
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
				h_avg /= count
				w_avg /= count
				#print (screen_height - h_avg, w_avg)
				point(res, h_avg, w_avg)
				points_to_fit_h.append(h_avg)
				points_to_fit_w.append(w_avg)
		# p = np.polyfit(points_to_fit_h, points_to_fit_w, 3)
		# for i in range(minimum, maximum):
		# 	line(res, i, evaluation(p, i))
			#print p(i)







#print "TEMP LIST: ", temp_list

# count = 0
# for i in range(screen_height):
# 	for j in range(screen_width):
# 		if visited[i][j] == -1:
# 			print "WRONG PAIR: ", i, j

# for j in range(screen_width):
# 	if visited[screen_height - 1][j] != 0:
# 		print visited[screen_height - 1][j], j


# for i in range(screen_height):
# 	for j in range(screen_width):
# 		if visited[i][j] == 1:
# 			res[i,j] = green
# 		elif visited[i][j] == 2:
# 			res[i,j] = white
# 		elif visited[i][j] == 3:
# 			res[i,j] = gray
# 		elif visited[i][j] == 4:
# 			res[i,j] = np.array([255,0,255])
# 		elif visited[i][j] == 5:
# 			res[i,j] = red
# 		elif visited[i][j] == 6:
# 			res[i,j] = cyan
# 		elif visited[i][j] == 7:
# 			res[i,j] = maroon


# for h in range(screen_height):
# 	print visited[h]

#point(res, 300, 300)
# for i in range(3):
# 	for j in range(3):
# 		res[300 + i,300 + j] = green

# temp_c = [100,100,100]
# for i in range(0,100):
# 	for j in range(100,400):
# 		#print out[i,j]
# 		res[i, j] = green
# 		#print out[i,j]

#temp = [[0,0],[1,1]]

cv2.imshow('imsage',res)
cv2.waitKey(0)

cv2.imwrite(outputFilename, res)
