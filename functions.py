import cv2
import math
import numpy as np

from matplotlib import pyplot as plt

def rotate_about_center(img, angle, scale=1.):
	w = img.shape[1]
	h = img.shape[0]
	rangle = np.deg2rad(angle)  # angle in radians
	# calculate new image width and height
	nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
	nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
	# get rotation matrix
	rot_mat = cv2.getRotationMatrix2D((nw/2, nh/2), angle, scale)
	# calculate the move from the old center to the new center combined
	# with the rotation
	rot_move = np.dot(rot_mat, np.array([(nw-w)/2, (nh-h)/2, 0]))
	# the move only affects the translation, so update the translation
	# part of the transform
	rot_mat[0, 2] += rot_move[0]
	rot_mat[1, 2] += rot_move[1]

	nw = int(math.ceil(nw))
	nh = int(math.ceil(nh))

	rotated = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))))

	cut_h = nh-h
	cut_w = nw-w

	if cut_h!=0 and cut_w!=0:
		#If crop necessary
		cropped = rotated[cut_h:-cut_h, cut_w:-cut_w]
	else:
		cropped = rotated

	return cropped