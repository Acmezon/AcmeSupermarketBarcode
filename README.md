# AcmeSupermarketBarcode
Aplicación AcmeSupermarketBarcode para el preprocesamiento, detección y decodificación de códigos de barras. Trabajo para la asignatura MC (Matemática Computacional) del Máster de Ingeniería Informática de la Universidad de Sevilla.

## Desarrollado por 
* Daniel de los Reyes Leal
* Alejandro Sánchez Medina

## Instalación
* Actualizar las bases de datos de repositorios
 * sudo apt-get update
* Instalar herramientas para compilar OpenCV 3.0:
 * sudo apt-get install build-essential cmake git pkg-config
* Instalar herramientas para leer formatos de imagen:
 * sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
* Instalar funcionalidades GUI de OpenCV 3.0:
 * sudo apt-get install libgtk2.0-dev
* Instalar paquetes para optimizar algunas funciones de OpenCV, como las operaciones con matrices:
 * sudo apt-get install libatlas-base-dev gfortran
* Instalar pip:
 * wget https://bootstrap.pypa.io/get-pip.py
 * sudo python3 get-pip.py
* Instalar herramientas de entornos virtuales de Python:
 * sudo pip3 install virtualenv virtualenvwrapper
* Configurar entorno virtual:
 * sudo nano ~/.bashrc
* Pegar las siguientes líneas de texto en el fichero:
 * export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
 * export WORKON_HOME=$HOME/.virtualenvs
 * source /usr/local/bin/virtualenvwrapper.sh
* Recargar el fichero de configuración:
 * source ~/.bashrc
* Instalar Python 3.4:
 * sudo apt-get install python3.4-dev
* Descargar los repositorios de OpenCV:
 * cd ~
 * git clone https://github.com/Itseez/opencv.git
 * cd opencv
 * git checkout 3.0.0
 * cd ~
 * git clone https://github.com/Itseez/opencv_contrib.git
 * cd opencv_contrib
 * git checkout 3.0.0
* Construir OpenCV:
 * cd ~/opencv
 * mkdir build
 * cd build
 * cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ..
 * sudo make -j4
 * sudo make install
 * sudo ldconfig
* Crear el entorno virtual
 * mkvirtualenv cv
 * workon cv
* Copiar OpenCV a nuestro entorno virtual de trabajo:
 * cd ~/.virtualenvs/cv/lib/python3.4/site-packages/
 * ln -s /usr/local/lib/python3.4/dist-packages/cv2.cpython-34m.so cv2.so
* Instalar numpy:
 * pip install numpy
* Instalar Matplotlib y Tkinter para gráficas de histogramas:
 * sudo apt-get install tcl-dev tk-dev python-tk python3-tk
 * sudo apt-get install python-matplotlib
 * sudo apt-get build-dep python-matplotlib
 * cd ~
 * git clone https://github.com/matplotlib/matplotlib.git
 * cd matplotlib
 * python setup.py install