# -*-coding:utf-8-*-
import collections
import cv2
import preproc
import math
import numpy as np
import matplotlib.pyplot as plt


def decode_image(path, function_threshold=0, blur_strength=(7, 7)):
    """
    Toma la imagen de un codigo de barras y lo decodifica, devolviendo
    un vector como el siguiente:
    [1, 2, 1, 3, 4]
    La interpretacion es que en la imagen del codigo existe una barra de
    grosor 1, luego un espacio de grosor 2, luego otra barra de grosor 1...

    Los grosores finales son independientes de la imagen original, ya que estan
    expresados en relacion al grosor base
    """

    image = preproc.run(path, blur_strength)

    # Se le aplica el detector de bordes de Canny
    canny = canny_edge(image, 200, 100)

    # Se inicializa un vector de rhos y thetas que iran guardando los
    # valores encontrados para los distintos umbrales
    rhos = np.array([])
    thetas = np.array([])

    # Se aplica Hough de forma iterativa aumentando el umbral y almacenando
    # las lineas que se han encontrado para cada uno. Esto se hace para tener
    # el numero de votos de cada linea
    threshold = 1
    while True:
        lines = cv2.HoughLines(
            canny, 1, np.pi / 90, threshold, min_theta=-(np.pi / 6),
            max_theta=np.pi / 6)
        if lines is None:
            break

        lines = np.ravel(lines)
        rhos = np.append(rhos, lines[::2])
        thetas = np.append(thetas, lines[1::2])

        threshold += 1

    t_votes = collections.Counter(thetas)
    most_common_theta = t_votes.most_common(1)[0][0]

    lines = np.fromiter(zip(rhos, thetas), dtype=[('r', 'i4'), ('t', 'f4')])
    lines = np.sort(lines[lines['t'] == most_common_theta], order=['r', 't'])

    rho_values, _ = zip(*lines)
    r_votes = collections.Counter(rho_values)

    rho_values = np.array([])
    votes = np.array([])

    # Se ordena el diccionario por la distancia de las lineas (rho)
    accum = collections.OrderedDict(
        sorted(r_votes.items(), key=lambda k: k[0]))

    rho_values, votes = zip(*accum.items())
    rho_values, votes = np.array(rho_values), np.array(votes)

    # Se filtran las lineas que no pasen un umbral y se llevan a 0 para
    # eliminar ruido
    high_votes = votes > function_threshold
    votes = votes[high_votes]
    rho_values = rho_values[high_votes]

    # Se extiende el array de distancias para que los valores cubran
    # todos los enteros de principio a fin
    rho_comp = np.arange(rho_values[0], rho_values[-1] + 1)
    votes_comp = np.zeros(rho_comp.shape)

    # Se mira en que posiciones del array original de valores se encontraban
    # los votos y se asignan en el nuevo array extendido. Esto se hace para no
    # perder los votos de las lineas originales
    votes_indices = np.searchsorted(rho_comp, rho_values)
    votes_comp[votes_indices] = votes

    non_zero = np.nonzero(votes_comp)[0]
    control_first = non_zero[0:4]
    control_end = non_zero[-4:]
    control_middle = non_zero[28:32]

    # Se obtiene el ancho base de una linea como la media del ancho de cada una
    # de las lineas de control, redondeando hacia arriba
    base_width = np.around(np.mean(
        np.concatenate((np.diff(control_first),
                        np.diff(control_middle),
                        np.diff(control_end)))))

    # Se obtiene el grosor de cada linea, relativo al grosor base, dividiendo
    # el ancho original por el base y redondeando hacia arriba
    lines_width = np.around(np.diff(np.nonzero(votes_comp)) / base_width)

    # TODO:
    # apuntar si alguna se queda en .5 para variarla hacia abajo si la
    # decodificacion falla

    # Se devuelve el resultado
    return lines_width[0]


def canny_edge(img, t_1, t_2, aperture=3, l2gradient=True):
    """
    Aplica el detector de bordes Canny a la imagen de entrada.
    Primero la binariza y luego aplica el detector
    """

    cv2.imshow('img', img)
    (_, thresh) = cv2.threshold(
        img, 230, 255, cv2.THRESH_BINARY)

    thresh = np.uint8(thresh)

    percentage = 100
    kernel_height = np.around(thresh.shape[0] * (percentage / 100))

    kernel = np.ones((kernel_height, 1), np.uint8)
    opened = cv2.dilate(thresh, kernel)
    opened = cv2.erode(opened, kernel)

    kernel = np.ones((1, 2), np.uint8)
    opened = cv2.dilate(opened, kernel)

    canny = np.copy(opened)
    cv2.Canny(opened, t_1, t_2, canny, aperture, l2gradient)

    return canny
