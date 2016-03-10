import cv2
import functions
import math
import numpy as np

def run(in_file, n=2):
    """
    Computes a inclination correction process with DFT transformation and Hough lines.
        Input:
            in_file: RGB Image input path.
            n: Process iterations. Default: 2.
        Output:
            Corrected image.
    """

    image = cv2.imread(in_file, 0)

    result = np.copy(image)
    for i in range(0, n):

        dft = cv2.dft(np.float32(result), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0],
                                                     dft_shift[:, :, 1]))

        (_, thresh) = cv2.threshold(
            magnitude_spectrum, 230, 255, cv2.THRESH_BINARY)

        thresh = np.uint8(thresh)

        lines = cv2.HoughLines(thresh, 1, np.pi / 180, 30)

        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            m_numerator = y2 - y1
            m_denominator = x2 - x1

            angle = np.rad2deg(math.atan2(m_numerator, m_denominator))
            result = functions.rotate_about_center(result, angle)

    return result