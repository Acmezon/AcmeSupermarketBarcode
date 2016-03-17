# -*-coding:utf-8-*-
import barcode_read
import math
import traceback
import translate_ean


def main():
    """
    1. Se asume que la imagen es la básica, sencilla (incluso sintética),
    se localiza perfectamente el codigo de barras, así que no se aplica
    DFT, se pone el filtro a 5x5 y se lanza

    2. Si falla, se sigue asumiendo imagen medianamente sencilla pero no
    tanto, aun DFT a 0, se varía el filtro de a 6, 7, 4 y 3

    3. Si todo lo anterior falla, DFT a 5, filtro a 5x5

    4. Si sigue fallando, DFT a 5, filtro a 6, 7, 4 y 3

    5. Si sigue fallando, DFT a 1, filtro a 5, 6, 7, 4 y 3

    """
    blur_strengths = [(5, 5), (6, 6), (7, 7), (4, 4), (3, 3)]
    inclination_ns = [0, 5, 1]
    success = False
    i = 0
    blur = i % len(blur_strengths)
    inclinations = math.floor(i / len(blur_strengths))

    number = -1
    while not success:
        if i >= len(inclination_ns) * len(blur_strengths):
            break

        print("Iteracion: {0}\nFuerza del emborronado:\
             {1}\nNumero de DFTs: {2}".format(
            i+1, blur_strengths[blur], inclination_ns[inclinations]))
        try:
            lines = barcode_read.decode_image(
                'resources/tatuaje.png', tuple(blur_strengths[blur]),
                inclination_ns[inclinations])
        except Exception:
            lines = None
            traceback.print_exc()
        i += 1
        blur = i % len(blur_strengths)
        inclinations = math.floor(i / len(blur_strengths))

        if lines is not None:
            print(lines)
            number = translate_ean.translate(lines)

            if number != -1:
                success = True

    print(number)
    return number

if __name__ == "__main__":
    main()
