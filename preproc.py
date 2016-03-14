import cv2
import dft
import functions
import math
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
    image = dft.run(in_file, 1)

    box, thresh = functions.find_contours(image, blur_strength)
    p1, p2 = box[0], box[1]

    m_numerator = p1[1] - p2[1]
    m_denominator = p1[0] - p2[0]

    angle = np.rad2deg(math.atan2(m_numerator, m_denominator))

    if angle != 90:
        rotated = functions.rotate_about_center(thresh, angle)

        _, thresh = functions.find_contours(rotated, blur_strength)

    return thresh
