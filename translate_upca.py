# -*-coding:utf-8-*-
import numpy as np

def translate_to_binary(lines_spaces_vector):
    """
    Translate lines-spaces vector to a binary code
    """
    r = []
    is_line = True

    if np.any(lines_spaces_vector > 7) or np.any(lines_spaces_vector < 0):
        raise ValueError('Error: Incorrect image bars/spaces transcription')
    else:
        for n in lines_spaces_vector:
            r = np.append(r, np.repeat(int(is_line), n))
            is_line = not is_line
    return r


def translate_byte(binary_code, side):
    """
    Translate a single 7-bits code to its number code format.
        Input:
            binary_code: Numpy array.
            side: int 0/1. 0: Left, 1: Right.
    """
    bin_string = ""

    for x in binary_code:
        bin_string += str(int(x))

    if (side == 0):
        left_dict = {
            '0001101': 0,
            '0011001': 1,
            '0010011': 2,
            '0111101': 3,
            '0100011': 4,
            '0110001': 5,
            '0101111': 6,
            '0111011': 7,
            '0110111': 8,
            '0001011': 9
        }
        return left_dict[bin_string]
    elif (side == 1):
        right_dict = {
            '1110010': 0,
            '1100110': 1,
            '1101100': 2,
            '1000010': 3,
            '1011100': 4,
            '1001110': 5,
            '1010000': 6,
            '1000100': 7,
            '1001000': 8,
            '1110100': 9
        }
        return right_dict[bin_string]
    else:
        raise ValueError("Side param incorrect.")


def checksum(barcode):
    """
    Computes a 1D barcode checksum
        Output:
            bool. Returns if checksum is correct.
    """
    # 1. Add the values of the digits in positions 1, 3, 5, 7, 9, and 11.
    term1 = int(barcode[0]) + int(barcode[2]) + int(barcode[4]) + \
        int(barcode[6]) + int(barcode[8]) + int(barcode[10])
    # 2. Multiply this result by 3.
    term2 = 3 * term1
    # 3. Add the values of the digits in positions 2, 4, 6, 8, and 10.
    term3 = int(barcode[1]) + int(barcode[3]) + \
        int(barcode[5]) + int(barcode[7]) + int(barcode[9])
    # 4. Sum the results of steps 2 and 3.
    term4 = term2 + term3
    # 5. The check character is the smallest number which, when added to the
    # result in step 4, produces a multiple of 10.
    term5 = ((10 - term4) % 10) % 10
    return str(term5) == barcode[-1]


def translate(vector):
    """
    Translate binary code to number code
      Input:
          vector: Lines and spaces size vector.
      Output:
          Barcode.
    """
    vector = translate_to_binary(vector)

    start_middle = 3 + 6 * 7
    end_middle = start_middle + 5

    frontier_array = np.array([1., 0., 1.])
    middle_array = np.array([0., 1., 0., 1., 0.])

    if len(vector) == 95 and \
            np.array_equal(vector[0:3], frontier_array) and \
            np.array_equal(vector[-3:], frontier_array) and \
            np.array_equal(vector[start_middle:end_middle],
                           middle_array):

        left_part = vector[3:start_middle]
        right_part = vector[end_middle:-3]

        translateL = ""
        translateR = ""

        for group in range(0, 6):
            left = left_part[group * 7: (group + 1) * 7]
            right = right_part[group * 7: (group + 1) * 7]

            # First bit of each part always 0/1.
            # Left odd parity. Right even parity.
            if left[0] == 0 and \
                    right[0] == 1 and \
                    np.sum(left) % 2 != 0 and \
                    np.sum(right) % 2 == 0:
                translateL += str(translate_byte(left, 0))
                translateR += str(translate_byte(right, 1))

            else:
                raise ValueError("Error: Group " + str(group) +
                      " wrong format. Parity not matching.")
                return -1

        barcode = translateL + translateR
        if checksum(barcode):
            return barcode
        else:
            raise ValueError('Error: Checksum failed.')
            return -1

    else:
        raise ValueError('Error: Wrong format. Incorrent barcode length or frontier bars not matching.')
        return -1
