# AcmeSupermarketBarcode
Aplicación AcmeSupermarketBarcode para el preprocesamiento, detección y decodificación de códigos de barras. Trabajo para la asignatura MC (Matemática Computacional) del Máster de Ingeniería Informática de la Universidad de Sevilla.

## Desarrollado por 
* Daniel de los Reyes Leal
* Alejandro Sánchez Medina

## Ejecución
* Para ejecutar la decodificación con una imagen concreta. (Se admiten formatos .jpg y .png)
 * <code>python3 main.py -p "RUTA_AL_ARCHIVO"</code>
* Para ejecutar la bateria de pruebas con las imágenes de la carpeta /resources
 * <code>python3 test.py</code>

## Instalación
* Actualizar las bases de datos de repositorios
 * <code>sudo apt-get update</code>
* Instalar herramientas para compilar OpenCV 3.0:
 * <code>sudo apt-get install build-essential cmake git pkg-config</code>
* Instalar herramientas para leer formatos de imagen:
 * <code>sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev</code>
* Instalar funcionalidades GUI de OpenCV 3.0:
 * <code>sudo apt-get install libgtk2.0-dev</code>
* Instalar paquetes para optimizar algunas funciones de OpenCV, como las operaciones con matrices:
 * <code>sudo apt-get install libatlas-base-dev gfortran</code>
* Instalar pip:
 * <code>wget https://bootstrap.pypa.io/get-pip.py</code>
 * <code>sudo python3 get-pip.py</code>
* Instalar herramientas de entornos virtuales de Python:
 * <code>sudo pip3 install virtualenv virtualenvwrapper</code>
* Configurar entorno virtual:
 * <code>sudo nano ~/.bashrc</code>
* Pegar las siguientes líneas de texto en el fichero:
 * <code>export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3</code>
 * <code>export WORKON_HOME=$HOME/.virtualenvs</code>
 * <code>source /usr/local/bin/virtualenvwrapper.sh</code>
* Recargar el fichero de configuración:
 * <code>source ~/.bashrc</code>
* Instalar Python 3.4:
 * <code>sudo apt-get install python3.4-dev</code>
* Descargar los repositorios de OpenCV:
 * <code>cd ~</code>
 * <code>git clone https://github.com/Itseez/opencv.git</code>
 * <code>cd opencv</code>
 * <code>git checkout 3.0.0</code>
 * <code>cd ~</code>
 * <code>git clone https://github.com/Itseez/opencv_contrib.git</code>
 * <code>cd opencv_contrib</code>
 * <code>git checkout 3.0.0</code>
* Construir OpenCV:
 * <code>cd ~/opencv</code>
 * <code>mkdir build</code>
 * <code>cd build</code>
 * <code>cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ..</code>
 * <code>sudo make -j4</code>
 * <code>sudo make install</code>
 * <code>sudo ldconfig</code>
* Crear el entorno virtual
 * <code>mkvirtualenv cv</code>
 * <code>workon cv</code>
* Copiar OpenCV a nuestro entorno virtual de trabajo:
 * <code>cd ~/.virtualenvs/cv/lib/python3.4/site-packages/</code>
 * <code>ln -s /usr/local/lib/python3.4/dist-packages/cv2.cpython-34m.so cv2.so</code>
* Instalar numpy:
 * <code>pip install numpy</code>
* Instalar Matplotlib y Tkinter para gráficas de histogramas:
 * <code>sudo apt-get install tcl-dev tk-dev python-tk python3-tk</code>
 * <code>sudo apt-get install python-matplotlib</code>
 * <code>sudo apt-get build-dep python-matplotlib</code>
 * <code>cd ~</code>
 * <code>git clone https://github.com/matplotlib/matplotlib.git</code>
 * <code>cd matplotlib</code>
 * <code>python setup.py install</code>

 ## Referencias
* Chai, D. and Hock, F. (2005.). “Locating and Decoding EAN-13 Barcodes from Images Captured by Digital Cameras”. 2005 5th International Conference on Information Communications & Signal Processing. Bangkok. pp.. 1595 - 1599.
* Belussi, Luiz F. F.; Hirata, Nina S. T. (2011). “Fast QR Code Detection in Arbitrarily Acquired Images”.  Graphics, Patterns and Images (Sibgrapi), 2011 24th SIBGRAPI Conference. IEEE. Maceio, Alagoas. pp. 281 - 288
* Meng Wang, Li-Na Li, Zhao-Xuan Yang. “Gabor Filtering-Based Scale and Rotation Invariance feature for 2D Barcode Region Detection”. 20I0 International Conference on Computer Application and System Modeling. IEEE. Tianjin, China. 2010. Vol 5. pp.. 34 - 37.
* Viola, Paul; Jones, Michael (2001). “Rapid object detection using a boosted cascade of simple features”. Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference. IEEE. Kauai, HI, USA. Vol. 1. pp. I-511 - I-518.
* Maragos, P. (2013). “Representations for Morphological Image Operators and Analogies with Linear Operators”. In Advances in Imaging and Electron Physics, vol. 177 (edited by P.W. Hawkes), pp. 45-187