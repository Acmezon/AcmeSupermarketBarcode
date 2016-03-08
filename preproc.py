# import the necessary packages
import collections
import cv2
import dft
import functions
import math
import numpy as np
import time

from PIL import Image

def current_milli_time():
    return time.time()

def run(in_file, out_file, blur_strength=(7, 7)):
    t = current_milli_time()
    dft.run(in_file, 'results/corrected.png')
    dft.run('results/corrected.png', 'results/corrected.png')
    print('Tarda ' + str(current_milli_time() - t) + ' s.')
    '''img = 'results/corrected.png'

    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # cv2.imshow("Gradient", gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, blur_strength)
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Blurred", blurred)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Closed", closed)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    # cv2.imshow("Eroded", closed)

    closed = cv2.dilate(closed, None, iterations=4)
    # cv2.imshow("Dilated", closed)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    _, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    mask = np.zeros_like(image)

    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(mask, [box], -1, (0, 255, 0), -1)

    out = np.zeros_like(image)
    in_border = np.where(mask == 255)

    out[in_border[0], in_border[1], in_border[2]] = \
        image[in_border[0], in_border[1], in_border[2]]

    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Out", out)
    cv2.waitKey(0)

    (_, thresh) = cv2.threshold(
        out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = np.uint8(thresh)

    canny = np.copy(thresh)
    cv2.Canny(thresh, 200, 100, canny, 3, True)



    rhos = np.array([])
    thetas = np.array([])

    threshold = 1
    while True:
        lines = cv2.HoughLines(
            canny, 1, np.pi / 45, threshold, min_theta=-(np.pi / 6),
            max_theta=np.pi / 6)
        if lines is None:
            break

        lines = np.ravel(lines)
        rhos = np.append(rhos, lines[::2])
        thetas = np.append(thetas, lines[1::2])

        threshold += 1

    accum = {}
    for i in range(len(rhos)):
        if (rhos[i], thetas[i]) in accum:
            accum[(rhos[i], thetas[i])] += 1
        else:
            accum[(rhos[i], thetas[i])] = 1

    accum = collections.OrderedDict(
        sorted(accum.items(), key=lambda k: k[0][0]))

    accum_values = collections.defaultdict(int)
    for k, v in accum.items():
        accum_values[k[1]] += v

    accum_values = np.array(
        list(accum_values.items()), dtype=[('x', 'f4'), ('y', 'i4')])

    theta = accum_values[np.argsort(accum_values, order=('y', 'x'))[-1]][0]

    print(theta)

    angle = -(np.rad2deg(theta))
    print(np.rad2deg(theta))

    out = functions.rotate_about_center(out, angle)

    x, y = np.nonzero(out)
    out = out[x.min():x.max() + 1, y.min():y.max() + 1]

    processed_img = Image.fromarray(out)
    processed_img.save(out_file)'''

if __name__ == "__main__":
    run('resources/borr_gir.png', 'results/out.jpg', (3, 3))
