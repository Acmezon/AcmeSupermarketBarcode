import numpy as np


def translate_to_binary(lines_spaces_vector):
    """
    Translate lines-spaces vector to a binary code
    """
    r = []
    is_line = True
    # print(lines_spaces_vector)

    if np.any(lines_spaces_vector > 7) or np.any(lines_spaces_vector < 0):
        print('Error: Incorrect image bars/spaces transcription')
    else:
        for n in lines_spaces_vector:
            r = np.append(r, np.repeat(int(is_line), n))
            is_line = not is_line
    return r


def translate_byte_left(binary_code, parity_digit, group):
    """
    Translate a single 7-bits code to its number code format.
        Input:
            binary_code: Numpy array.
            side: int 0/1. 0: Left, 1: Right.
    """
    bin_string = ""

    for x in binary_code:
        bin_string += str(int(x))

    digit_dict = {
        0: 'LLLLLL',
        1: 'LLGLGG',
        2: 'LLGGLG',
        3: 'LLGGGL',
        4: 'LGLLGG',
        5: 'LGGLLG',
        6: 'LGGGLL',
        7: 'LGLGLG',
        8: 'LGLGGL',
        9: 'LGGLGL'
    }

    translate_dict = {
        'L': {
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
        },
        'G': {
            '0100111': 0,
            '0110011': 1,
            '0011011': 2,
            '0100001': 3,
            '0011101': 4,
            '0111001': 5,
            '0000101': 6,
            '0010001': 7,
            '0001001': 8,
            '0010111': 9
        }
    }

    letter = digit_dict[parity_digit][group]
    return translate_dict[letter][bin_string]


def translate_byte_right(binary_code):
    """
    Translate a single 7-bits code to its number code format.
        Input:
            binary_code: N,umpy array.
            side: int 0/1. 0: Left, 1: Right.
    """
    bin_string = ""

    for x in binary_code:
        bin_string += str(int(x))

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


def checksum(barcode):
    """
    Computes a 1D barcode checksum
        Output:
            bool. Returns if checksum is correct.
    """
    checksum = 0
    barcode_c = barcode[0:-1]
    for i, digit in enumerate(reversed(barcode_c)):
        checksum += int(digit) * 3 if (i % 2 == 0) else int(digit)

    term5 = (10 - (checksum % 10)) % 10
    return str(term5) == barcode[-1]


def get_parity_digit(parity):
    parity_dict = {
        'OOOOOO': 0,
        'OOEOEE': 1,
        'OOEEOE': 2,
        'OOEEEO': 3,
        'OEOOEE': 4,
        'OEEOOE': 5,
        'OEEEOO': 6,
        'OEOEOE': 7,
        'OEOEEO': 8,
        'OEEOEO': 9
    }
    return parity_dict[parity]


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
        parity = ""

        for group in range(0, 6):
            left = left_part[group * 7: (group + 1) * 7]
            right = right_part[group * 7: (group + 1) * 7]

            # First bit of each part always 0/1.
            # Left odd parity. Right even parity.
            if left[0] == 0 and \
                    right[0] == 1 and \
                    np.sum(right) % 2 == 0:

                if np.sum(left) % 2 != 0:
                    parity += "O"
                else:
                    parity += "E"
            else:
                print("Error: Group " + str(group) +
                      " wrong format. Parity not matching.")
                return -1

        parity_digit = get_parity_digit(parity)

        for group in range(0, 6):
            left = left_part[group * 7: (group + 1) * 7]
            right = right_part[group * 7: (group + 1) * 7]

            translateL += str(translate_byte_left(left, parity_digit, group))
            translateR += str(translate_byte_right(right))

        barcode = str(parity_digit) + translateL + translateR

        if checksum(barcode):
            return barcode
        else:
            print("Error: Checksum failed.")
            return -1

    else:
        print(
            "Error: Wrong format. Incorrent barcode \
            length or frontier bars not matching.")
        return -1
