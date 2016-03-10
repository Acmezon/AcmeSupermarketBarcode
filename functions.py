import cv2
import math
import numpy as np


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
    return cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))))