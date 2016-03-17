# -*-coding:utf-8-*-
import barcode_read
import collections
import math
import numpy as np
from shutil import copyfile
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
    dirname = './resources/'
    results_f = 'results.txt'
    dst_folder = './results/'
    # Crea o vacia el fichero, si ya existia
    with open(results_f, 'w') as f:
        pass

    blur_strengths = [(5, 5), (6, 6), (7, 7), (4, 4), (3, 3)]
    inclination_ns = [0, 5, 1]
    times = []
    config = []
    results = []

    for fn in os.listdir(dirname):
        print(fn)
        success = False
        i = 0
        blur = 0
        inclinations = 0

        number = -1
        t = time.time()
        while not success:
            if i >= len(inclination_ns) * len(blur_strengths):
                break

            """
            print("Iteracion: {0}\nFuerza del emborronado:\
                 {1}\nNumero de DFTs: {2}".format(
                i + 1, blur_strengths[blur], inclination_ns[inclinations]))
            """
            try:
                lines = barcode_read.decode_image(
                    dirname + fn, tuple(blur_strengths[blur]),
                    inclination_ns[inclinations])
            except Exception:
                lines = None
                # traceback.print_exc()
            i += 1
            blur = i % len(blur_strengths)
            inclinations = math.floor(i / len(blur_strengths))

            if lines is not None:
                try:
                    number = translate_ean.translate(lines)
                    success = True
                except Exception:
                    success = False

        elapsed = time.time() - t
        times.append(elapsed)
        results.append(success)

        with open(results_f, 'a') as f:
            config_1 = config_2 = "-"
            if success:
                blur = (i - 1) % len(blur_strengths)
                inclinations = math.floor((i - 1) / len(blur_strengths))
                config_1 = blur_strengths[blur]
                config_2 = inclination_ns[inclinations]
                config.append(i - 1)

            f.write("\n{0} {1} {2} {3} {4}".format(
                fn, elapsed, int(success), config_1, config_2))

        print(number)

    with open(results_f, 'a') as f:
        f.write("\n\n")
        mean_time = np.mean(times)
        proportion = np.count_nonzero(results) / len(results)

        configs = collections.Counter(config)

        f.write("\nAciertos: {0}".format(proportion * 100))
        f.write("\nTiempo: {0}".format(np.sum(times)))
        f.write("\nTiempo medio: {0}".format(mean_time))
        f.write("\nMejor tiempo: {0}".format(np.amin(times)))
        f.write("\nPeor tiempo: {0}".format(np.amax(times)))
        f.write("\nMejores configuraciones:")
        f.write("\n\t{0}".format(configs))

if __name__ == "__main__":
    main()
