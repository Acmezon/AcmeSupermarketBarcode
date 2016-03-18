# -*-coding:utf-8-*-
import collections
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import preproc

from decimal import Decimal, ROUND_HALF_UP


def get_lines_width(lines, start, end, iteration):
    """
    Get lines and converts to run-length encoding.
    """
    lines = lines[start:end]

    control_first = lines[0:3]
    control_end = lines[-3:]
    control_middle = lines[27:31]

    myround = np.vectorize(
        lambda x: Decimal(Decimal(x).
                          quantize(Decimal('1'), rounding=ROUND_HALF_UP)))

    # Se obtiene el ancho base de una linea como la media del ancho de cada una
    # de las lineas de control, redondeando hacia arriba
    base_width = int(myround(
        np.mean(np.hstack((control_first, control_middle, control_end)))))

    # Se obtiene el grosor de cada linea, relativo al grosor base, dividiendo
    # el ancho original por el base y redondeando hacia arriba
    lines_width = myround(np.array(lines) / (base_width - 0.1 * iteration))

    return lines_width


def decode_image(path, blur_strength=(7, 7),
                 inclination_n=0):
    """
    Toma la imagen de un codigo de barras y lo decodifica, devolviendo
    un vector como el siguiente:
    [1, 2, 1, 3, 4]
    La interpretacion es que en la imagen del codigo existe una barra de
    grosor 1, luego un espacio de grosor 2, luego otra barra de grosor 1...

    Los grosores finales son independientes de la imagen original, ya que estan
    expresados en relacion al grosor base
    """

    image = preproc.run(path, blur_strength, inclination_n)

    image = cv2.bitwise_not(image)

    sample = image[image.shape[0] / 2, :]

    plt.subplot(1, 2, 1)
    plt.plot(sample)

    sample_mean = np.mean(sample)
    sample[np.where(sample < sample_mean)] = 0
    sample[np.where(sample >= sample_mean)] = 1

    generated_barcode = np.zeros((500, sample.shape[0]))
    generated_barcode[:, np.where(sample == 1)] = 255
    generated_barcode = 255 - generated_barcode

    # cv2.imwrite("generated_barcode.jpg", generated_barcode)

    non_zero = np.nonzero(sample)
    first_non_zero, last_non_zero = non_zero[0][0], non_zero[0][-1] + 1

    if last_non_zero < sample.shape[0]:
        sample = np.delete(sample, np.s_[last_non_zero:])

    if first_non_zero > 0:
        sample = np.delete(sample, np.s_[0:first_non_zero])

    """
    plt.subplot(1, 2, 2)
    plt.plot(sample)
    plt.show()
    """

    pos, = np.where(np.diff(sample) != 0)
    pos = np.concatenate(([0], pos + 1, [len(sample)]))
    lines = [b - a for (a, b) in zip(pos[:-1], pos[1:])]

    diff_2 = np.diff(np.abs(np.diff(lines)))

    combinations = [[0, 0], [0, 1], [1, 0], [1, 1]]

    max_sum = 0
    combination_res = None
    lines_res = None
    for combination in combinations:
        start, end = np.where(diff_2 == combination[0])[0][0], np.where(
            diff_2 == combination[1])[0][-1]
        lines_width = get_lines_width(lines, start, end + 1 + 2, 0)
        # +1 porque no se incluye el final, +2 porque cada diff reduce el array
        # en original en 1 posicion

        lines_sum = np.sum(lines_width)
        if lines_sum == 95:
            lines_res = lines_width
            break
        elif lines_sum > max_sum:
            max_sum = lines_sum
            combination_res = combination
            lines_res = lines_width

    """
    if 90 < max_sum and max_sum < 95:
        i = 0
        while True:
            start = np.where(diff_2 == combination_res[0])[0][0]
            end = np.where(diff_2 == combination_res[1])[0][-1]
            lines_width = get_lines_width(lines, start, end + 1 + 2, i)
            i += 1

            if np.sum(lines_width) >= 95:
                lines_res = lines_width
                break
    elif 95 < max_sum and max_sum < 99:
        i = 0
        while True:
            start = np.where(diff_2 == combination_res[0])[0][0]
            end = np.where(diff_2 == combination_res[1])[0][-1]
            lines_width = get_lines_width(lines, start, end + 1 + 2, -i)
            i += 1

            if np.sum(lines_width) <= 95:
                lines_res = lines_width
                break
    """

    if np.sum(lines_res) != 95:
        lines_res = None

    return lines_res
