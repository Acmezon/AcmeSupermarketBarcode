import cv2
import math
import numpy as np

from PIL import Image


def run(in_file, out_file):
    img = cv2.imread(in_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape

    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0],
                                                 dft_shift[:, :, 1]))

    (_, thresh) = cv2.threshold(
        magnitude_spectrum, 230, 255, cv2.THRESH_BINARY)

    thresh = np.uint8(thresh)

    lines = cv2.HoughLines(thresh, 1, np.pi / 180, 30)
    magnitude_spectrum_lines = np.copy(magnitude_spectrum)

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

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

        if cols > rows:
            out_dims = (cols, cols)
        else:
            out_dims = (rows, rows)

        rotated_img = cv2.warpAffine(img, M, out_dims)

        cv2.line(magnitude_spectrum_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    b, g, r = cv2.split(rotated_img)
    rotated_img = cv2.merge([r, g, b])
    rotated_img = Image.fromarray(rotated_img)
    rotated_img.save(out_file)

    magnitude_spectrum = Image.fromarray(magnitude_spectrum).convert('RGB')
    magnitude_spectrum.save('results/fourier.png')

    magnitude_spectrum_lines = Image.fromarray(
        magnitude_spectrum_lines).convert('RGB')
    magnitude_spectrum_lines.save('results/fourier_lines.png')
