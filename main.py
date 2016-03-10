# -*-coding:utf-8-*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import hough_line


def main():
    image = cv2.imread('resources/barcode.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    out, angles, d = hough_line(gray)

    angles = angles * 100

    angles = angles.astype(int)

    angles = angles / 100

    angle = np.where(angles == 0)[0][1]
    ys = out[:, angle]

    fig, ax = plt.subplots()

    # the histogram of the data
    ax.bar(d, ys)

    plt.xlabel('rho')
    plt.ylabel('height')
    plt.title('90º Angle')
    plt.show()

if __name__ == "__main__":
    main()
