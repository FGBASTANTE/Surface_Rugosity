# -*- coding: utf-8 -*-
"""
Fichero de inicialización de parámetros del módulo principal.
También contiene algunas funciones
@author: Fernando
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import matplotlib.cm as cm
import numpy as np

# Evitar que aparezcan los plots en pantalla
PLT_OFF = True

# Ruta al directorio de los raw data
RAW_FOLDER = "C:/Users/Fernando/Duro/Fernando/A_Google_Drive/Python_Scripts/Tilt_test/I.Perez_Cuarcita"

# Apellido que indica su formato (.xyz en este caso) de los ficheros raw
APELLIDO = '.xyz'

# Nombre del subdiretorio donde se guardan los resultados (debe existir)
SAVE_SUBDIR = "Cuarcita"

# Nombre del fichero excel donde se guardarán los resultados
res_nam = "resultados"

# Guardarlos en disco duro. Si False el fichero queda en memoria
USE_WRITER = False

# Crear las figuras 2D
MAKE_PLOTS = True

# Crear la figura con los raw data (consume mucho tiempo)
PLOT_RAW = False

# Ver una pequeña zona en 3D (las coordenadas en el script principal)
PLOT_3D = False

# Para interactuar con Mayavi (tiene que estar instalado):
USE_MAYAVI = False

# Proporción lineal (ventana) de la zona central a analizar (con punto decimal)
PROP = 1.0

# Espaciado del grid (x/y) de trabajo en unidades del sistema de coordenadas
STEP = 16

# True para eliminar la tendencia de z con un polinomio de grado DEG
USE_DETREND = False
DEG = 7

# True para filtrar las Z's  (n_stdz (max/min) desviaciones estándar en sript)
FILTER_Z = False

def detrend_fit(x, y, z, deg):
    """ Función para eliminar la tendencia de una superficie z=z(x, y)
     utilizando un polinomio de grado deg en x e y """
    from numpy.polynomial import polynomial

    # Normalizo el grid para que no den valores descomunales
    rg_x, rg_y = 0.5*(np.max(x) - np.min(x)), 0.5*(np.max(y) - np.min(y))
    x_n = np.asarray(x)/rg_x
    y_n = np.asarray(y)/rg_y
    z = np.asarray(z)

    deg_L = [deg, deg]
    vander = polynomial.polyvander2d(x_n, y_n, deg_L)
    c = np.linalg.lstsq(vander, z, rcond=None)[0]
    c = c.reshape(deg+1, -1)
    z = np.polynomial.polynomial.polyval2d(x_n, y_n, c)
    return z, c, rg_x, rg_y

def fig_roseta(ang, frecang, anf, filename_ros):
    """ se crea la roseta de distribución direcciones de máxima pendiente"""
    fig_p = plt.figure(figsize=(3, 3))
    fig_p.suptitle('Steepest Slope Direction\n Rose Diagram')

    #Se define el origen del eje y su tamaño (ancho y altura en proporción)
    ax = fig_p.add_axes([0.2, 0.2, 0.6, 0.55], polar=True)
    ax.bar(ang[0:-1], frecang, width=anf, bottom=0.0)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=np.arange(0, 360, 45))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rgrids([])
    ax.set_title('')
    fig_p.savefig(filename_ros, dpi=300)

def fig_hist(z, xlab="", ylab="", titt="", filename="hist"):
    """ se crea el histograma"""
    fig_p = plt.figure(figsize=(2.5, 2.5))
    fig_p.suptitle(titt)
    ax = fig_p.add_axes([0.35, 0.2, 0.6, 0.6])
    ax.hist(z, bins=256)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab, style='oblique')
    ax.set_title('')
    namsav = filename + titt
    fig_p.savefig(namsav, dpi=600)

def fig_grid(grid, titt="titt", bar_lab="bar_", extent=None, filename="grid"):
    fig, ax = plt.subplots(constrained_layout=False)
    fig.suptitle(titt)
    xlab = r'$X  (\mu m)$'
    ylab = r'$Y  (\mu m)$'
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    pos_fig = ax.imshow(grid, extent=extent, aspect='equal')
    bar_fig = fig.colorbar(pos_fig, ax=ax)
    bar_fig.ax.set_ylabel(bar_lab, labelpad=0, y=0.5)
    fig_nam = titt + filename
    fig.savefig(fig_nam, dpi=300)

def fig_grid2D(grid2D, titt, extension, filename):
    """ se crea la imagen hillshade de un objeto 2D, vg grid_z1"""
    fig_p = plt.figure(figsize=(5, 5))
    fig_p.suptitle('Raw Data (hillshade image)\n\nz exaggerattion x 10')

    # Añado el eje y limito su tamaño para que quepan los numericos
    ax = fig_p.add_axes([0.2, 0.2, 0.6, 0.6])
    ax.set_xlabel(r'$X  (\mu m)$')
    ax.set_ylabel(r'$Y  (\mu m)$', labelpad=-0.2)

    # Se crea una array de ceros, y se le añade el max y el min
    # Es un artificio para poder dibujar el colobar
    tru = np.zeros_like(grid2D)
    tru[0, 0], tru[0, 1] = np.min(grid2D), np.max(grid2D)

    # Use a proxy artist for the colorbar...se crea y elimina
    im = ax.imshow(tru, cmap=cm.gist_earth, extent=extension)
    im.remove()

    # La iluminación
    ls = LightSource(azdeg=-20, altdeg=45)

    # Y finalmente la imagen
    ax.imshow(ls.shade(grid2D, cmap=cm.gist_earth,\
    vert_exag=10, blend_mode='hsv', vmin=-60, vmax=25), extent=extension)

    # Fraction y pad son para obtener una buena disposición del colobar
    barf = plt.colorbar(im, fraction=0.046, pad=0.04)
    barf.ax.set_ylabel(r'$Z  (\mu m)$', labelpad=0, y=0.5)

    fig_nam = titt + filename
    fig_p.savefig(fig_nam, dpi=300)
    