import cv2
import dft
import functions
import numpy as np

from matplotlib import pyplot as plt

def run(in_file, blur_strength=(7, 7), inclination_n=4):
    """
    Preprocessing of the barcode image
        Input:
            in_file: RGB Image input filepath.
            blur_strength: Average blurring mask size. Default: (7,7)
        Output:
            Processed image
    """
    image = dft.run(in_file, 4)

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

    """plt.subplot(4,2,1), plt.imshow(image ,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(4,2,2), plt.imshow(gradient ,cmap = 'gray')
    plt.title('Gradient'), plt.xticks([]), plt.yticks([])
    plt.subplot(4,2,3), plt.imshow(thresh ,cmap = 'gray')
    plt.title('Threshold'), plt.xticks([]), plt.yticks([])
    plt.subplot(4,2,4), plt.imshow(closed ,cmap = 'gray')
    plt.title('Closed'), plt.xticks([]), plt.yticks([])"""

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    _, cnts, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    mask = np.zeros_like(image)

    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(mask, [box], -1, (255, 255, 255), -1)

    out = np.zeros_like(image)
    in_border = np.where(mask == 255)

    out[in_border[0], in_border[1]] = image[in_border[0], in_border[1]]

    (_, thresh2) = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    x, y = np.nonzero(out)
    thresh2 = out[x.min():x.max() + 1, y.min():y.max() + 1]

    """plt.subplot(4, 2, 5), plt.imshow(closed, cmap='gray')
    plt.title('closed'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 6), plt.imshow(mask, cmap='gray')
    plt.title('Mask'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 7), plt.imshow(out, cmap='gray')
    plt.title('Out threshold'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 8), plt.imshow(thresh2, cmap='gray')
    plt.title('Out + Cropped'), plt.xticks([]), plt.yticks([])

    plt.show()"""

    return thresh2
