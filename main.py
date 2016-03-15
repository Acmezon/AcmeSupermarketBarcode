# -*-coding:utf-8-*-
import barcode_read
import numpy as np
import translate


def main():
    lines = barcode_read.decode_image(
        'resources/test_3.jpg', blur_strength=(5, 5))

    number = translate.translate(lines)
    print(number)

if __name__ == "__main__":
    main()
