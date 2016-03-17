# -*-coding:utf-8-*-
import cv2
import math
import numpy as np

from matplotlib import pyplot as plt

def rotate_about_center(img, angle, scale=1.):
    """
    Rotates a image about its centre with correct crop.
        Input:
            img: Input image
            angle: Angle in degrees.
            scale: Re-scaling. Default:1
        Outout:
            Rotated & cropped image
    """
    img = cv2.bitwise_not(img)
    w = img.shape[1]
    h = img.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # get rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) / 2, (nh - h) / 2, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    nw = int(math.ceil(nw))
    nh = int(math.ceil(nh))

    rotated = cv2.warpAffine(
        img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))))

    cut_h = int(math.ceil((nh - h) / 2))
    cut_w = int(math.ceil((nw - w) / 2))

    if cut_h != 0 and cut_w != 0:
        # If crop necessary
        cropped = rotated[cut_h:-cut_h, cut_w:-cut_w]
    else:
        cropped = rotated
    return 255 - rotated


def get_roi(image, blur_strength=(7, 7)):
    """
    Detect region of interest (barcode) and crop the input image to adjust to it.
        Input:
            image: Input image.
            blur_strength = Average filter mask size.
        Output:
            Cropped image with the barcode.
    """
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, blur_strength)

    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # closing operation. kernel 15x15
    kernel = np.ones((20, 20), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    """
    plt.subplot(4, 2, 1), plt.imshow(image, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 2), plt.imshow(gradient, cmap='gray')
    plt.title('Gradient'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 3), plt.imshow(thresh, cmap='gray')
    plt.title('Threshold'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 4), plt.imshow(closed, cmap='gray')
    plt.title('Closed'), plt.xticks([]), plt.yticks([])
    """

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    _, cnts, _ = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    mask = np.zeros_like(image)

    columns = box[:, 0]

    _, ncols = mask.shape

    if columns[0] > 10:
        columns[0:2] = columns[0:2] - 10
    else:
        columns[0:2] = 0

    if columns[-1] < ncols - 10:
        columns[-2::] = columns[-2::] + 10
    else:
        columns[-2::] = ncols

    box[:, 0] = columns

    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(mask, [box], -1, (255, 255, 255), -1)

    mask_neg = cv2.bitwise_not(mask)

    out = np.zeros_like(image)
    in_border = np.where(mask == 255)

    out[in_border[0], in_border[1]] = image[in_border[0], in_border[1]]

    out_c = out + mask_neg

    x, y = np.nonzero(out)
    out_c = out_c[x.min():x.max() + 1, y.min():y.max() + 1]

    """
    plt.subplot(4, 2, 5), plt.imshow(closed, cmap='gray')
    plt.title('closed'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 6), plt.imshow(mask, cmap='gray')
    plt.title('Mask'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 7), plt.imshow(out, cmap='gray')
    plt.title('Out threshold'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 8), plt.imshow(out_c, cmap='gray')
    plt.title('Out + Cropped'), plt.xticks([]), plt.yticks([])

    plt.show()
    """

    return box, out_c


def get_barcode(image, blur_strength=(7, 7)):
    _, out_c = get_roi(image, blur_strength)

    return out_c
