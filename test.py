# -*-coding:utf-8-*-
import collections
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from mpl_toolkits.mplot3d import Axes3D


def current_milli_time():
    return time.time()


func_threshold = 55
t = current_milli_time()

image = cv2.imread('resources/barcode.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(_, thresh) = cv2.threshold(
    gray, 230, 255, cv2.THRESH_BINARY)

thresh = np.uint8(thresh)

canny = np.copy(thresh)
cv2.Canny(thresh, 200, 100, canny, 3, True)
cv2.imshow('Canny', canny)

rhos = np.array([])
thetas = np.array([])

threshold = 1
while True:
    lines = cv2.HoughLines(
        canny, 1, np.pi / 45, threshold, min_theta=0, max_theta=np.pi / 2)
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

fig, ax = plt.subplots()

rho_values = np.array([])
votes = np.array([])

accum = collections.OrderedDict(sorted(accum.items(), key=lambda k: k[0][0]))

for key, val in accum.items():
    if key[1] == 0.0:
        rho_values = np.append(rho_values, np.array([key[0]]))
        votes = np.append(votes, np.array([val]))

print(current_milli_time() - t)

high_votes = votes > func_threshold
votes = votes[high_votes]
rho_values = rho_values[high_votes]

"""
rho_comp = np.arange(rho_values[0], rho_values[-1] + 1)
votes_comp = np.zeros(rho_comp.shape)

votes_indices = np.searchsorted(rho_comp, rho_values)
votes_comp[votes_indices] = votes

# the histogram of the data
ax.plot(rho_comp, votes_comp, '-o')
plt.xlabel('rho')
plt.ylabel('height')
plt.title('90º Angle')
plt.show()
# cv2.imshow("Lines", image)
# cv2.waitKey(0)
"""
