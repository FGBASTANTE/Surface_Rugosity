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
plt.ioff()

# Se introduce la ruta al directorio de los raw data
DATA_FOLDER = Path("E:/Tilt_test/I.Perez_Granito")

# Se lee el nombre de los ficheros de puntos (terminan en .xyz)
FICHEROS = [files for files in os.listdir(DATA_FOLDER) if files.endswith(".xyz")]

# Decomentar para hacer el análisis de solo algún fichero
FICHEROS = [FICHEROS[1]]

# Nombre del fichero excel donde se guardarán los resultados
RES_NAM = "resultados_UTb.xlsx"

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
    rg_x, rg_y = 0.5*(np.max(x) - np.min(x)), 0.5*(np.max(y) - np.min(y))
    x = np.asarray(xo)/rg_x
    y = np.asarray(yo)/rg_y
    z = np.asarray(zo)
    _deg = [deg, deg]
    vander = polynomial.polyvander2d(x, y, _deg)
    c = np.linalg.lstsq(vander, z, rcond=None)[0]
    c = c.reshape(deg+1, -1)
    z = np.polynomial.polynomial.polyval2d(x, y, c)
    return z

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

    # Decomentar para eliminar la tendencia de z con un polinomio de grado DEG
    USE_DETREND = True
    DEG = 7

    # Solo se aplica si se desea eliminar la tendencia de z
    # Se intercambia la z sin tendencia con la zo original
    if USE_DETREND:
        zo_detrend = detrend_fit(xo, yo, zo, deg=DEG)
#        zoo = zo.copy()
        zo = zo - zo_detrend
        points = np.c_[points, zo, zo_detrend]
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

    # Si se filtran las Z's se utilizan los límites siguientes en función de la sqo
    n_stdzo = 4
    lsupZ, linfZ = meanzo + n_stdzo * stdzo, meanzo - n_stdzo * stdzo

    # Se calculan las diferencias de coordenadas para ver el intervalo de muestreo
    # Los puntos están alineados con el eje Y en sentido descendiente
    gradxo = np.diff(xo)
    gradyo = np.min(np.diff(yo))

    # Proporción lineal (ventana) de la zona central a analizar (con el punto decimal)
    prop = 1.0
    umbrinx = xmino + rangxo * (1- prop)/2
    umbrsux = umbrinx + prop *rangxo
    umbriny = ymino + rangyo * (1- prop)/2
    umbrsuy = umbriny + prop *rangyo

    # Filtro para obtener la ventana de estudio y la opción de filtrar las Z's
    FILTER_Z = False

    # set mask for x,y filtering
    mask = (points[:, 0] > umbrinx) & (points[:, 0] < umbrsux) \
            & (points[:, 1] > umbriny) & (points[:, 1] < umbrsuy)

    # set mask for z filtering
    if FILTER_Z:
        mask = mask & (points[:, 2] > linfZ) & (points[:, 2] < lsupZ)

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

    # Defino el paso y límites (disminuyo unos steps en los bordes) del grid
    step = 16
    step2 = step * step
    _xmin, _xmax = xmin + 6 * step, xmax - 6 * step
    _ymin, _ymax = ymin + 6 * step, ymax - 6 * step

    # y creo el grid (se invierte el orden y el signo del step en el eje Y
    # para mantener la dirección y el sentido de los ejes)
    grid_y, grid_x = np.mgrid[_ymax:_ymin:-step, _xmin:_xmax:step]
    extension = (np.min(grid_x), np.max(grid_x), np.min(grid_y), np.max(grid_y))

    # Interpolo Z
    grid_z1 = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # La interpolación puede introducir valores nan (en bordes), se enmascaran
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

    # For plotting hillshade: parámetros para el plotting
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

    # Quitamos el apellido al filename (las hojas de excel aceptan hasta 31 char.)
    filename = filename.replace('.xyz', '')
 
    if USE_DETREND:
        filename = filename + "UT"

    # Paso los resultados a un dataframe y los guardo en el excel
    pdsave = pd.DataFrame.from_dict(results, orient='index')
    pdsave.to_excel(WRITER, sheet_name=filename)

    # Constante par definir si se crean las figuras
    Make_Plots = False

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
        ap = aspectrad[~aspectrad.mask].ravel()
        aspectrad = 90 - ap * 180 / np.pi
        smagrad = umask_smagrad * 180 / np.pi
        pos61 = ax6[0].hist(aspectrad, bins=256)
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
        ax7[0].set_ylabel(r'$Data number$')
        ax7[0].set_title('z distribution density')

        ls = LightSource(azdeg=-20, altdeg=45)
        pos72 = ax7[1].imshow(ls.shade(grid_z1, cmap=plt.cm.gist_earth,\
        vert_exag=10, blend_mode='hsv', vmin=-32, vmax=20))

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
        frecang = pos61[0]/np.sum(pos61[0])
        ang = pos61[1]*np.pi/180
        angp = np.mean(ang)*180/np.pi
        anf = np.diff(ang)
        bars = ax.bar(ang[0:-1], frecang, width=anf, bottom=0.0)
        ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Se guarda
        filenamefig5 = str(filename) + "_P_" + str(prop) +"_roseta.png"
        fig5.savefig(filenamefig5, dpi=2400)


        # Se cierran las figuras (con plt.ioff no es necesario pero...)
        plt.close('all')

WRITER.save()
WRITER.close()








# Utilidades

    #    import mplstereonet
    #    fig, ax = mplstereonet.subplots()
    #    strikes = aspectrad
    #    dips = smagrad
    #    estereo = mplstereonet.kmeans(strikes, dips)
    #    cax = ax.density_contourf(strikes, dips, measurement='poles')
    #    ax.pole(strikes, dips)
    #    ax.grid(True)
    #    fig.colorbar(cax)




#from mpl_toolkits import mplot3d as mpl
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(grid_x, grid_y, gradz_y, 50, cmap='binary')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#
##ax = plt.axes(projection='3d')
#ax.plot_surface(grid_x, grid_y, grid_z1, rstride=1, cstride=1,
#                cmap='Greys', edgecolor='none')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z');
#ax.set_title('surface');
#
#np.savetxt('puntos3D.txt', fichero, delimiter=';') #se guarda
