# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:10:09 2019

Módulo que lee ficheros de puntos (X, Y, Z) no estructurados y
calcula una serie de parámetros indicadores de la rugosidad.
Para ello se crea un grid, y se interpola el valor de las Z`s en dicho grid.

@author: Fernando García Bastante
Universidad de Vigo

TODO
multithreading
"""
# Se importan los módulos que se van a utilizar

import os
from pathlib import Path
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.interpolate import griddata
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
from openpyxl import load_workbook

# Evitamos que se generen plots por defecto: consumen recursos
#plt.ioff()

# Se introduce la ruta al directorio de los raw data
DATA_FOLDER = Path("E:/Tilt_test/I.Perez_Caliza")

# Y el apellido que indica su formato (terminan en .xyz en este caso)
APELLIDO = '.xyz'

# Se lee el nombre de los ficheros de puntos
FICHEROS = [files for files in os.listdir(DATA_FOLDER) \
           if files.endswith(APELLIDO)]

# Parámetro para definir si se crean las figuras
Make_Plots = False

# Algunos parámetros por defecto para los plots (figsize en pulgadas)
plt.rcParams['figure.figsize'] = [6.0, 4.0]
plt.rcParams['figure.dpi'] = 100

# Parámetro para ver una pequeña zona en 3D
PLOT_3D = False

# coordenadas de la zona de visualización
Min_Cx, Max_Cx = 1500, 3000
Min_Cy, Max_Cy = 2500, 5000

# Para interactuar con Mayavi:
USE_MAYAVI = False

# Proporción lineal (ventana) de la zona central a analizar (con punto decimal)
prop = 1.0

# Espaciado del grid (x/y) de trabajo en unidades del sistema de coordenadas
step = 16

# True para eliminar la tendencia de z con un polinomio de grado DEG
USE_DETREND = True
DEG = 7

# True para filtrar las Z's a partir n_stdz (max y min) desviaciones estándar
FILTER_Z = False
n_stdzomax, n_stdzomin = 4, 100

# Nombre del fichero excel donde se guardarán los resultados
RES_NAM = "pruebas"

# Lo modificamos en caso de quitar tendencia y/o utilizar filtro
if USE_DETREND: RES_NAM = RES_NAM + '_UT'
if FILTER_Z: RES_NAM = RES_NAM + '_F'
RES_NAM = RES_NAM + '.xlsx'

# Para guardar el fichero excel en disco duro
# Si no se guarda el fichero queda en memoria
USE_WRITER = True

# Decomentar para hacer el análisis de solo algún fichero
FICHEROS = [FICHEROS[5]]

# Se abre el fichero para guardar resultados (o se crea si no existe)
# Si existe NO SOBREESCRIBE las hojas
try:
    book = load_workbook(RES_NAM)
    WRITER = pd.ExcelWriter(RES_NAM, engine='openpyxl')
    WRITER.book = book
    WRITER.sheets = dict((ws.title, ws) for ws in book.worksheets)
    print(WRITER.sheets)
except:
    WRITER = pd.ExcelWriter(RES_NAM)

def detrend_fit(x, y, z, deg):
    """ Función para eliminar la tendencia de una superficie z=z(x, y)
     utilizando un polinomio de grado deg en x e y """
    from numpy.polynomial import polynomial

    # Normalizo el grid para que no den valores descomunales
    rg_x, rg_y = 0.5*(np.max(x) - np.min(x)), 0.5*(np.max(y) - np.min(y))
    x = np.asarray(x)/rg_x
    y = np.asarray(y)/rg_y
    z = np.asarray(z)

    _deg = [deg, deg]
    vander = polynomial.polyvander2d(x, y, _deg)
    c = np.linalg.lstsq(vander, z, rcond=None)[0]
    c = c.reshape(deg+1, -1)
    z = np.polynomial.polynomial.polyval2d(x, y, c)
    return z, c, rg_x, rg_y

# Comienzan los cálculos
for nick in FICHEROS:
    # Nombre del archivo a analizar
    filename = nick
    # Abro y leo todos los datos
    with open(DATA_FOLDER / filename, 'r') as pfile:
        points = pfile.read()
        # los convierto en números; los delimitadores son tabulaciones
        points = np.genfromtxt(StringIO(points))
    #opción B
    #points = np.loadtxt(nick)

    # Nº puntos y se extraen las coordenadas
    npoints, dim = points.shape[0], points.shape[1]
    xo, yo, zo = [points[:, i] for i in np.arange(dim)]

    # Calculamos algunos estadísticos de las Z's
    meanzo, stdzo = np.mean(zo), np.std(zo) # stdz es sqo

    # Si se filtran las Z's se utilizan los límites siguientes:
    lsupZ, linfZ = meanzo + n_stdzomax * stdzo, meanzo - n_stdzomin * stdzo

    # set mask for z filtering
    if FILTER_Z:
        mask = (points[:, 2] > linfZ) & (points[:, 2] < lsupZ)
        points = points[mask]
        xo, yo, zo = [points[:, i] for i in np.arange(dim)]

    # Se aplica si se desea eliminar la tendencia de z
    # Se intercambia la z sin tendencia con la zo original
    if USE_DETREND:
        zo_trend, coef_pol, rg_x, rg_y = detrend_fit(xo, yo, zo, deg=DEG)
        zo = zo - zo_trend

    # Se concatenan y reordenan (x, y, zo-z_detrend, zo, z_detrend)
        points = np.c_[points, zo, zo_trend]
        points = points[:, [0, 1, 3, 2, 4]]

    # Dimensiones espaciales, coordenadas, rango e intervalos
    xmino, xmaxo = np.min(xo), np.max(xo)
    rangxo = xmaxo - xmino
    ymino, ymaxo = np.min(yo), np.max(yo)
    rangyo = ymaxo - ymino
    extensiono = [xmino, xmaxo, ymino, ymaxo]
    zmino, zmaxo = np.min(zo), np.max(zo)
    rangzo = zmaxo - zmino

    # Calculamos algunos estadísticos de las Z's
    meanzo, stdzo = np.mean(zo), np.std(zo) # stdz es sqo
    sao = np.mean(np.abs(zo - meanzo))
    ssko = skew(zo)
    skuo = kurtosis(zo, fisher=False)

    # Se calculan las diferencias de coordenadas para ver el intervalo de muestreo
    # Los puntos están alineados con el eje Y en sentido descendiente
    gradxo = np.diff(xo)
    gradyo = np.min(np.diff(yo))

    # Límites de la ventana central a analizar
    umbrinx = xmino + rangxo * (1- prop)/2
    umbrsux = umbrinx + prop *rangxo
    umbriny = ymino + rangyo * (1- prop)/2
    umbrsuy = umbriny + prop *rangyo

    # set custom mask for x,y filtering
    mask = (points[:, 0] > umbrinx) & (points[:, 0] < umbrsux) \
            & (points[:, 1] > umbriny) & (points[:, 1] < umbrsuy)

    # Get the masked points
    xyz_filtered = points[mask]

    # Separo de nuevo las coordenadas de los puntos, ahora en la ventana
    x, y, z = [xyz_filtered[:, i] for i in np.arange(dim)]
    xmin, xmax = np.min(x), np.max(x)
    rangx = xmax - xmin
    ymin, ymax = np.min(y), np.max(y)
    rangy = ymax - ymin
    zmin, zmax = np.min(z), np.max(z)
    rangz = zmax - zmin
    rangz_x = rangz/rangx
    rangz_y = rangz/rangy

    # Límites (disminuyo unos steps en los bordes) del grid
    step2 = step * step
    _xmin, _xmax = xmin + 6 * step, xmax - 6 * step
    _ymin, _ymax = ymin + 6 * step, ymax - 6 * step

    # y creo el grid (se invierte el orden y el signo del step en el eje Y
    # para mantener la dirección y el sentido de los ejes)
    grid_y, grid_x = np.mgrid[_ymax:_ymin:-step, _xmin:_xmax:step]
    extension = (np.min(grid_x), np.max(grid_x), np.min(grid_y), np.max(grid_y))

    # Tendencia en el grid
    if USE_DETREND:
        g_x = np.asarray(grid_x[0])/rg_x
        g_y = np.asarray(grid_y[:, 0])/rg_y
        grid_zT = np.polynomial.polynomial.polygrid2d(g_x, g_y, coef_pol)

    # Interpolo Z
    grid_z1 = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # La interpolación puede introducit valores nan (en bordes), se enmascaran
    grid_z1 = np.ma.array(grid_z1, mask=np.isnan(grid_z1)) # Use a mask

    # Media, según los ejes (axis=0: media por columnas: x's fijos en cada una)
    # np.mean no tiene en cuenta los valores null
    meanZ_x, meanZ_y = np.mean(grid_z1, axis=0), np.mean(grid_z1, axis=1)

    # Se calculan los estadísticos eliminando los valores enmascarados
    # La array se aplana al desenmascarar (no es necesario utilizar ravel())
    umask_grid_z1 = grid_z1[~grid_z1.mask]
    n = umask_grid_z1.size
    meanz, stdz = np.mean(umask_grid_z1), np.std(umask_grid_z1) # Esto es sq

    ssk = skew(umask_grid_z1, None)
    sku = kurtosis(umask_grid_z1, None, fisher=False)
    sa = np.mean(np.abs(umask_grid_z1 - meanz))

    # Gradientes: primero en dirección vertical, luego en la horizontal
    # Gradient usa: (fn+1 - fn-1)/(2 x step))
    gradz_y, gradz_x = np.gradient(grid_z1, step)

    # Al cuadrado, quito la máscara y calculo el rsm, el valor absoluto, etc.
    grad2_x, grad2_y = np.square(gradz_x), np.square(gradz_y)
    grad2_xy = (grad2_x + grad2_y)  # magnitude of slope in radians

    umask_grad2_x, umask_grad2_y = grad2_x[~grad2_x.mask], grad2_y[~grad2_y.mask]
    umask_grad2_xy = grad2_xy[~grad2_xy.mask]
    nx, ny, nxy = umask_grad2_x.size, umask_grad2_y.size, umask_grad2_xy.size

    rsmz_x, rsmz_y = np.sqrt(np.mean(umask_grad2_x)), np.sqrt(np.mean(umask_grad2_y))
    rsmz_xy = np.sqrt(np.mean(umask_grad2_xy))

    # Cálculo del valor medio absoluto de la pendiente (no desenmascaro)
    modz_x, modz_y = np.mean(np.abs(gradz_x)), np.mean(np.abs(gradz_y))

    # For plotting hillshade: parámetros para el plotting propio
    az, elev = -20, 45
    azRad, elevRad = (360 - az + 90)*np.pi/180, (90-elev)*np.pi/180

    # Dirección de máxima pendiente (hacia el E y el N positiva) y buzamiento
    aspectrad = np.arctan2(gradz_y, -gradz_x) # Angle of aspect
    smagrad = np.arctan(np.sqrt(grad2_xy))  # Magnitude of slope in radians

    # Media y desviación estándar de la slope
    umask_smagrad = smagrad[~smagrad.mask]
    smagradmean = np.mean(umask_smagrad)
    smagradstd = np.std(umask_smagrad)

    # Función para el cálculo de la iluminación
    hs = ((np.cos(elevRad) * np.cos(smagrad)) + \
          (np.sin(elevRad)* np.sin(smagrad) * np.cos(azRad - aspectrad)))

    # Repito con las diferencias de primer orden (se pierde un dato)
    # Para dirección vertical axis=0, el sentido es i+1 - i, o sea, al revés
    slopez_y, slopez_x = np.diff(grid_z1, axis=0)/step, np.diff(grid_z1, axis=1)/step
    ravez_x, ravez_y = slopez_x[~slopez_x.mask], slopez_y[~slopez_y.mask]

    # Distancia real frente a la proyectada
    distz_x = np.mean(np.sqrt(1 + np.square(ravez_x)))
    distz_y = np.mean(np.sqrt(1 + np.square(ravez_y)))

    # Incremento de superficie/distancia
    sdr = np.mean(np.sqrt(1 + umask_grad2_xy) - 1)
    sdrx = np.mean(np.sqrt(1 + umask_grad2_x) - 1)
    sdry = np.mean(np.sqrt(1 + umask_grad2_y) - 1)

    # Unos cuantos estadísticos más
    meanz_x, meanz_y = np.mean(ravez_x), np.mean(ravez_y)
    stdz_x, stdz_y = np.std(ravez_x), np.std(ravez_y)

    meanz_px, meanz_nx = np.mean(ravez_x[ravez_x > 0]), np.mean(ravez_x[ravez_x <= 0])
    meanz_py, meanz_ny = np.mean(ravez_y[ravez_y > 0]), np.mean(ravez_y[ravez_y <= 0])

    mod1z_x = np.mean(np.abs(ravez_x))
    mod1z_y = np.mean(np.abs(ravez_y))

    # se calculan las segundas derivadas
    slopez_xx, slopez_yy = np.diff(slopez_x, axis=1)/step, np.diff(slopez_y, axis=0)/step
    ravez_xx, ravez_yy = slopez_xx[~slopez_xx.mask], slopez_yy[~slopez_yy.mask]

    # Unos cuantos estadísticos más
    meanz_xx, meanz_yy = np.mean(ravez_xx), np.mean(ravez_yy)
    stdz_xx, stdz_yy = np.std(ravez_xx), np.std(ravez_yy)

    # Pit and peak proportion (slope sign changes)
    ds_x = np.mean((np.diff(np.sign(slopez_x), axis=1) != 0)*1)
    ds_y = np.mean((np.diff(np.sign(slopez_y), axis=0) != 0)*1)

    # La mitad peaks y la otra mitad pits; la longitud de onda media será:
    l_x = step / (ds_x/2)
    l_y = step / (ds_y/2)

    # Diccionario con los resultados que queremos guardar
    results = {
        'fichero': filename,
        'filter' : FILTER_Z,
        'step' : step,
        'proporción' : prop,
        'points number' : n,
        'rangex' : rangx,
        'rangey' : rangy,
        'rangz' : rangz,
        'rangz_x' : rangz_x,
        'rangz_y' : rangz_y,
        'meanz' : meanz,
        'sa' : sa,
        'sq' : stdz,
        'ssk' : ssk,
        'sku' : sku,
        'meanz_x' : meanz_x,
        'meanz_y' : meanz_y,
        'meanz_px' : meanz_px,
        'meanz_nx' : meanz_nx,
        'meanz_py' : meanz_py,
        'meanz_ny' : meanz_ny,
        'stdz_x' : stdz_x,
        'stdz_y' : stdz_y,
        'modz_x' : modz_x,
        'modz_y' : modz_y,
        'mod1z_x' : mod1z_x,
        'mod1z_y' : mod1z_y,
        'rsmz_x' : rsmz_x,
        'rsmz_y' : rsmz_y,
        'rsmz_xy' : rsmz_xy,
        'distz_x' : distz_x,
        'distz_y' : distz_y,
        'meanz_xx' : meanz_xx,
        'meanz_yy' : meanz_yy,
        'stdz_xx' : stdz_xx,
        'stdz_yy' : stdz_yy,
        'l_x' : l_x,
        'l_y' : l_y,
        'rangzo' : rangzo,
        'meanzo' : meanzo,
        'sao' : sao,
        'sqo' : stdzo,
        'ssko' : ssko,
        'skuo' : skuo,
        'smagradmean' : smagradmean,
        'smagradstd' : smagradstd,
        'sdr' : sdr,
        'sdrx' : sdrx,
        'sdry' : sdry
        }

    # Quitamos el apellido al filename (las hojas excel aceptan hasta 31 char.)
    filename = filename.replace('.xyz', '')

    if USE_DETREND:
        filename = filename + "UT"

    # Paso los resultados a un dataframe y se guardan en el excel
    pdsave = pd.DataFrame.from_dict(results, orient='index')
    pdsave.to_excel(WRITER, sheet_name=filename)

    # Defino algunas variables utilizadas en algunos gráficos
    ap = aspectrad[~aspectrad.mask].ravel()
    aspectgrad = 90 - ap * 180 / np.pi
    aspect_hist, aspect_hist_bin = np.histogram(aspectgrad, bins=256)
    frecang = aspect_hist/np.sum(aspect_hist)
    ang = aspect_hist_bin*np.pi/180
    angp = np.mean(ang)*180/np.pi
    anf = np.diff(ang)

    if Make_Plots:
        # Gráficos (se crean varias figuras con plantilla nrows x ncols)
        fig1, (ax1, ax2) = plt.subplots(figsize=(10, 15),\
               nrows=2, ncols=2, constrained_layout=False)
        fig1.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,\
                             hspace=0.25, wspace=0.35)
        fig1.suptitle(filename)

        # Crear esta figura consume mucho tiempo, decomentar si se desea
        # Z's originales
    #    pos11 = ax1[0].scatter(xo, yo, c=zo, vmin=zmino, vmax=zmaxo)
    #    ax1[0].set_xlabel(r'$X  (\mu m)$')
    #    ax1[0].set_ylabel(r'$Y  (\mu m)$')
    #    ax1[0].set_title('raw data')
    #    bar11 = fig1.colorbar(pos11, ax=ax1[0])
    #    bar11.ax.set_ylabel(r'$Z  (\mu m)$', labelpad=0, y=0.5)
    #    ax1[0].axis('equal')

        # Z's interpoladas
        pos12 = ax1[1].imshow(grid_z1, extent=extension, aspect='equal')
        ax1[1].set_xlabel(r'$X  (\mu m)$')
        ax1[1].set_ylabel(r'$Y  (\mu m)$')
        ax1[1].set_title('interpolate data')
        bar12 = fig1.colorbar(pos12, ax=ax1[1])
        bar12.ax.set_ylabel(r'$Z  (\mu m)$')

        # Z's gradient y
        pos21 = ax2[0].imshow(gradz_y, extent=extension)
        ax2[0].set_xlabel(r'$X  (\mu m)$')
        ax2[0].set_ylabel(r'$Y  (\mu m)$')
        ax2[0].set_title('gradient y data')
        bar21 = fig1.colorbar(pos21, ax=ax2[0])
        bar21.ax.set_ylabel(r'$Z grad_y$')

        # Z's gradient x
        pos22 = ax2[1].imshow(gradz_x, extent=extension)
        ax2[1].set_xlabel(r'$X  (\mu m)$')
        ax2[1].set_ylabel(r'$Y  (\mu m)$')
        ax2[1].set_title('gradient x data')
        bar22 = fig1.colorbar(pos22, ax=ax2[1])
        bar22.ax.set_ylabel(r'$Z grad_x$')

        # Se guarda
        filenamefig1 = str(filename) + "_P_" + str(prop) + "_imagrd" + ".png"
        fig1.savefig(filenamefig1, dpi=1200)


        fig2, (ax3, ax4) = plt.subplots(figsize=(10, 15),\
               nrows=2, ncols=2, constrained_layout=False)
        fig2.suptitle(filename)
        fig2.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                             wspace=0.35)

        # Z's gradient y (primera diferencia)
        pos31 = ax3[0].imshow(slopez_y, extent=extension)
        ax3[0].set_xlabel(r'$X  (\mu m)$')
        ax3[0].set_ylabel(r'$Y  (\mu m)$')
        ax3[0].set_title('gradient y data')
        bar31 = fig2.colorbar(pos31, ax=ax3[0])
        bar31.ax.set_ylabel(r'$Z grad_y$')

        # Z's gradient x (primera diferencia)
        pos32 = ax3[1].imshow(slopez_x, extent=extension)
        ax3[1].set_xlabel(r'$X  (\mu m)$')
        ax3[1].set_ylabel(r'$Y  (\mu m)$')
        ax3[1].set_title('gradient x data')
        bar32 = fig2.colorbar(pos32, ax=ax3[1])
        bar32.ax.set_ylabel(r'$Z grad_x$')

        # Texture in x and y direction
        pos411 = ax4[0].imshow(grad2_xy, cmap='coolwarm')
        pos412 = ax4[0].imshow(hs, cmap='gray', alpha=0.5)
        ax4[0].invert_yaxis()
        ax4[0].set_xlabel('X')
        ax4[0].set_ylabel('Y')
        ax4[0].set_title('Textura')
        ax4[0].invert_yaxis()

        # Aspect angle
        pos42 = ax4[1].imshow(aspectrad, extent=extension)
        ax4[1].set_xlabel(r'$X  (\mu m)$')
        ax4[1].set_ylabel(r'$Y  (\mu m)$')
        ax4[1].set_title('Aspect angle')
        bar42 = fig2.colorbar(pos42, ax=ax4[1])
        bar42.ax.set_ylabel(r'rad')

        # Se guarda
        filenamefig2 = str(filename) + "_P_" + str(prop) + "_imagrdtext" + ".png"
        fig2.savefig(filenamefig2, dpi=1200)

           
        fig3, (ax5, ax6) = plt.subplots(figsize=(15, 10),\
               nrows=2, ncols=2, constrained_layout=False)
        fig3.suptitle(filename)
        fig3.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                             wspace=0.35)

        # x Slope hist
        pos51 = ax5[0].hist(ravez_x, bins=256, range=(-1., 1.))
        ax5[0].set_xlabel(r'$x Slope$')
        ax5[0].set_ylabel(r'$Data number$')
        ax5[0].set_title('x Slope hist')

        # y Slope hist
        pos52 = ax5[1].hist(ravez_y, bins=256, range=(-1., 1.))
        ax5[1].set_xlabel(r'$y Slope$')
        ax5[1].set_ylabel(r'$Data number$')
        ax5[1].set_title('y Slope hist')

        # Aspect angle hist
        smagrad = umask_smagrad * 180 / np.pi
        pos61 = ax6[0].hist(aspectgrad, bins=256)
        ax6[0].set_xlabel(r'$Direction (ºN)$')
        ax6[0].set_ylabel(r'$Data number$')
        ax6[0].set_title('Aspect angle hist')

        # Z`s trends in x and y direction
        pos611 = ax6[1].plot(grid_y[:, 0], meanZ_y, 'r')
        pos612 = ax6[1].plot(grid_x[0, :], meanZ_x)
        ax6[1].set_xlabel('X (blue) or Y (red)')
        ax6[1].set_ylabel('z')
        ax6[1].set_title('Tendencias')

        # Se guarda
        filenamefig3 = str(filename) + "_P_" + str(prop) + "_hist" + ".png"
        fig3.savefig(filenamefig3, dpi=1200)


        fig4, (ax7, ax8) = plt.subplots(figsize=(15, 10),\
               nrows=2, ncols=2, constrained_layout=False)
        fig4.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
                             hspace=0.25, wspace=0.35)
        fig4.suptitle(filename)

        # z hist
        pos71 = ax7[0].hist(z, bins=256)
        ax7[0].set_xlabel(r'$z$')
        ax7[0].set_ylabel(r'$Data_number$')
        ax7[0].set_title('z distribution density')

        # Variantes de hs
        ls = LightSource(azdeg=-20, altdeg=45)
        pos72 = ax7[1].imshow(ls.shade(grid_z1, cmap=plt.cm.gist_earth,\
        vert_exag=10, blend_mode='hsv', vmin=-32, vmax=20), extent=extension)

        ls = LightSource(azdeg=70, altdeg=45)
        pos81 = ax8[0].imshow(ls.shade(grid_z1, cmap=plt.cm.gist_earth,\
        vert_exag=10, blend_mode='hsv', vmin=-32, vmax=20))

        ls = LightSource(azdeg=-20, altdeg=45)
        pos821 = ax8[1].imshow(grad2_xy, cmap='coolwarm', vmin=-32, vmax=20)
        pos822 = ax8[1].imshow(hs, cmap='gray', alpha=0.5)

        # Se guarda
        filenamefig4 = str(filename) + "_P_" + str(prop) +"_hisZ.png"
        fig4.savefig(filenamefig4, dpi=2400)

        # Polar diagram
        fig5 = plt.figure(figsize=(8, 8))
        fig5.suptitle(filename)
        ax = fig5.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        bars = ax.bar(ang[0:-1], frecang, width=anf, bottom=0.0)
        ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Se guarda
        filenamefig5 = str(filename) + "_P_" + str(prop) +"_roseta.png"
        fig5.savefig(filenamefig5, dpi=2400)

        # Se cierran las figuras (con plt.ioff no es necesario pero...)
        plt.close('all')

if USE_WRITER:
    WRITER.save()
    WRITER.close()

# Utilidades

def fig_grid2D(grid2D):
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
    im = ax.imshow(tru, cmap=plt.cm.gist_earth, extent=extension)
    im.remove()

    # La iluminación
    ls = LightSource(azdeg=-20, altdeg=45)

    # Y finalmente la imagen
    ax.imshow(ls.shade(grid2D, cmap=plt.cm.gist_earth,\
    vert_exag=10, blend_mode='hsv', vmin=-60, vmax=25), extent=extension)

    # Fraction y pad son para obtener una buena disposición del colobar
    barf = plt.colorbar(im, fraction=0.046, pad=0.04)
    barf.ax.set_ylabel(r'$Z  (\mu m)$', labelpad=0, y=0.5)

    # Se guarda
    filename = "Raw_data"
    if USE_DETREND:
        filename = filename + "UT"

    fig_p.savefig(filename, dpi=1200)
    return

def fig_hist(z):
    """ se crea el histograma"""
    fig_p = plt.figure(figsize=(2.5, 2.5))
    fig_p.suptitle('z Probability Density\n ')
    ax = fig_p.add_axes([0.35, 0.2, 0.6, 0.6])

    ax.hist(z, bins=256)
    ax.set_xlabel(r'$z (\mu m)$')
    ax.set_ylabel('data number', style='oblique')
    ax.set_title('')

    fig_p.savefig('histZ', dpi=1200)
    return

def fig_roseta():
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

    fig_p.savefig('roseta', dpi=1200)
    return

if PLOT_3D:
    from mpl_toolkits import mplot3d as mpl

    # Se crea la máscara y se seleccionas los puntos a dibujar
    m_points = (points[:, 0] < Max_Cx) & (points[:, 0] > Min_Cx) \
                & (points[:, 1] < Max_Cy) & (points[:, 1] > Min_Cy)
    p_points = points[m_points]

    # Se crea el plot
    fig_3D, ax1 = plt.subplots(figsize=(5, 5))
    ax1 = fig_3D.gca(projection='3d')
    ax1.plot_trisurf(p_points[:, 0], p_points[:, 1], p_points[:, 2],\
                     antialiased=True, cmap=plt.cm.gist_earth)
    ax1.set_xlabel(r'$X  (\mu m)$')
    ax1.set_ylabel(r'$Y  (\mu m)$')
    ax1.set_zlabel(r'$Z  (\mu m)$')

if USE_MAYAVI:
    from mayavi import mlab

    # Por ejemplo:
    mlab.surf(grid_x, grid_y, grid_z1)
