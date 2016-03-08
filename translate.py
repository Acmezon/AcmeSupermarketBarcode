import numpy as np


def translate_to_binary(lines_spaces_vector):
    """
    Translate lines-spaces vector to a binary code
    """
    r = []
    is_line = True

    if len(np.where(np.logical_or(lines_spaces_vector < 1,
                                  lines_spaces_vector > 4))[0]):
        print("Error")

    for n in lines_spaces_vector:
        r = np.append(r, np.repeat(int(is_line), n))
        is_line = not is_line

    return r


def translate(vector):
    """
    Translate binary code to number code
      Input:
          vector: Lines and spaces size vector.
      Output:
          Barcode.
    """
    vector = translate_to_binary(vector)

    # np.array_equal(vector[39:44],np.array([0.,1.,0.,1.,0.])
    if len(vector) == 95 and \
            np.array_equal(vector[0:3], np.array([1., 0., 1.])) and \
            np.array_equal(vector[-3:], np.array([1., 0., 1.])) and \
            np.array_equal(vector[3 + 6 * 7:(3 + 6 * 7) + 5],
                           np.array([0., 1., 0., 1., 0.])):

        left_part = vector[3:39]
        right_part = vector[44:-3]

        for group in range(0, 6):
            left = left_part[group * 7: (group + 1) * 7]
            right = right_part[group * 7: (group + 1) * 7]

            # First bit of each part always 0/1.
            # Left odd parity. Right even parity.
            if left[0] == 0 and \
                    right[0] == 1 and \
                    np.sum(left) % 2 != 0 and \
                    np.sum(right) % 2 == 0:
                print("Good " + str(group + 1))
            else:
                print("Wrong: " + str(group + 1))

    else:
        print("Wrong format")

vector = np.array([1, 1, 1, 3, 2, 1, 1, 1, 4, 1, 1, 1, 1, 1, 4, 3, 2, 1, 1, 3, 2, 1, 1, 3, 2, 1, 1,
                   1, 1, 1, 1, 1, 2, 1, 2, 2, 3, 1, 1, 2, 2, 2, 2, 1, 1, 1, 3, 2, 1, 2, 3, 1, 2, 1, 2, 2, 1, 1, 1, ])
y = translate(vector)
