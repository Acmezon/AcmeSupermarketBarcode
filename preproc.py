import cv2
import dft
import functions


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

    barcode = functions.get_barcode(image, blur_strength)
    #cv2.imshow("barcode", barcode)

    return barcode
