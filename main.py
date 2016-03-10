# -*-coding:utf-8-*-
import barcode_read
import translate

def main():
    lines = barcode_read.decode_image('resources/barcode_gir_5.jpg', blur_strength=(3,3))

    #print(lines)

    #number = translate.translate(lines)

    #print(number)

if __name__ == "__main__":
    main()
