# -*-coding:utf-8-*-
import collections
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from mpl_toolkits.mplot3d import Axes3D


def current_milli_time():
    return time.time()


function_threshold = 55

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

high_votes = votes > function_threshold
votes = votes[high_votes]
rho_values = rho_values[high_votes]

bars_start = rho_values[::2]
bars_end = rho_values[1::2]

bars_width = bars_end - bars_start

rho_comp = np.arange(rho_values[0], rho_values[-1] + 1)
votes_comp = np.zeros(rho_comp.shape)

votes_indices = np.searchsorted(rho_comp, rho_values)
votes_comp[votes_indices] = votes

max_height = np.amax(votes_comp)
high_lines = votes_comp > (max_height - 10)
votes_comp[high_lines] = max_height

min_height = np.amin(votes_comp[np.nonzero(votes_comp)])
# print(min_height)

low_lines = np.where(votes_comp < (min_height + 15))
votes_comp[np.intersect1d(low_lines, np.nonzero(votes_comp))] = min_height

high_lines = np.where(votes_comp == max_height)
control_first = high_lines[0][0:4]
control_end = high_lines[0][-4:]
control_middle = high_lines[0][8:12]

base_width = np.around(np.mean(
    np.concatenate((np.diff(control_first),
                    np.diff(control_middle),
                    np.diff(control_end)))))

lines_width = np.around(np.diff(np.nonzero(votes_comp)) / base_width)

print(lines_width)

print(current_milli_time() - t)

bars = np.nonzero(votes_comp)
# the histogram of the data
ax.plot(rho_comp, votes_comp, '-o')
plt.xlabel('rho')
plt.ylabel('height')
plt.title('90º Angle')
plt.show()
# cv2.imshow("Lines", image)
# cv2.waitKey(0)
