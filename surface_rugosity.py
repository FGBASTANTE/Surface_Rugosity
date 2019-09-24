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

from io import StringIO
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import kurtosis, skew
import pandas as pd
from openpyxl import load_workbook
from ini_surface_rugosity import fig_roseta, fig_hist, fig_grid, fig_grid2D, \
    detrend_fit
from ini_surface_rugosity import RAW_FOLDER, APELLIDO, SAVE_SUBDIR, MAKE_PLOTS
from ini_surface_rugosity import PLOT_RAW, PLOT_3D, USE_MAYAVI, PROP, STEP, \
    USE_DETREND, USE_WRITER, DEG, FILTER_Z, res_nam, PLT_OFF

# Evitar que se muestren plots: consumen recursos
if PLT_OFF:
    plt.ioff()

# Ruta al directorio de los raw data
raw_folder = Path(RAW_FOLDER)

# Se leen los nombres de los ficheros de puntos
FICHEROS = [files for files in os.listdir(raw_folder) \
           if files.endswith(APELLIDO)]

# Ruta del directorio para guardar los resultados
save_dir = os.path.join(os.getcwd(), SAVE_SUBDIR)

# Se modifica el result filename cuando se quita tendencia y/o utilizar filtro
if USE_DETREND:
    res_nam = res_nam + '_UT'
if FILTER_Z:
    res_nam = res_nam + '_F'

# Algunos parámetros por defecto para los plots (figsize en pulgadas)
if MAKE_PLOTS:
    plt.rcParams['figure.figsize'] = [6.0, 4.0]
    plt.rcParams['figure.dpi'] = 100

# coordenadas de la zona de visualización
if PLOT_3D:
    Min_Cx, Max_Cx = 1500, 3000
    Min_Cy, Max_Cy = 2500, 5000

# Filtrar las Z's a partir n_stdz (max y min) desviaciones estándar
if FILTER_Z:
    n_stdzomax, n_stdzomin = 4, 100

# Decomentar para hacer el análisis de solo algún fichero
#FICHEROS = [FICHEROS[5]]

# Ruta y fichero para guardar los resultados
file_path = os.path.join(save_dir, res_nam + '.xlsx')

# Se abre el fichero para guardar resultados (o se crea si no existe)
try:
    book = load_workbook(file_path)
    writer = pd.ExcelWriter(file_path, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    print(writer.sheets)
except:
    print("Se creará un fichero de resultados nuevo")
    writer = pd.ExcelWriter(file_path)

# Comienzan los cálculos
for nick in FICHEROS:
    # Nombre del archivo a analizar
    filename = nick

    # Abro y leo todos los datos
    with open(raw_folder / filename, 'r') as pfile:
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

    # set mask for z filtering
    if FILTER_Z:
        # Si se filtran las Z's se utilizan los límites siguientes:
        lsupZ, linfZ = meanzo + n_stdzomax * stdzo, meanzo - n_stdzomin * stdzo
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

    # Se calculan las diferencias de coordenadas para ver el intervalo de
    # muestreo. Los puntos están alineados con el eje Y en sentido descendiente
    gradxo = np.diff(xo)
    gradyo = np.min(np.diff(yo))

    # Límites de la ventana central a analizar
    umbrinx = xmino + rangxo * (1- PROP)/2
    umbrsux = umbrinx + PROP *rangxo
    umbriny = ymino + rangyo * (1- PROP)/2
    umbrsuy = umbriny + PROP *rangyo

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
    _xmin, _xmax = xmin + 6 * STEP, xmax - 6 * STEP
    _ymin, _ymax = ymin + 6 * STEP, ymax - 6 * STEP

    # y creo el grid (se invierte el orden y el signo del step en el eje Y
    # para mantener la dirección y el sentido de los ejes)
    grid_y, grid_x = np.mgrid[_ymax:_ymin:-STEP, _xmin:_xmax:STEP]
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
    # Gradient usa: (fn+1 - fn-1)/(2 x STEP))
    gradz_y, gradz_x = np.gradient(grid_z1, STEP)

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
    slopez_y, slopez_x = np.diff(grid_z1, axis=0)/STEP, np.diff(grid_z1, axis=1)/STEP
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
    slopez_xx, slopez_yy = np.diff(slopez_x, axis=1)/STEP, np.diff(slopez_y, axis=0)/STEP
    ravez_xx, ravez_yy = slopez_xx[~slopez_xx.mask], slopez_yy[~slopez_yy.mask]

    # Unos cuantos estadísticos más
    meanz_xx, meanz_yy = np.mean(ravez_xx), np.mean(ravez_yy)
    stdz_xx, stdz_yy = np.std(ravez_xx), np.std(ravez_yy)

    # Pit and peak proportion (slope sign changes)
    ds_x = np.mean((np.diff(np.sign(slopez_x), axis=1) != 0)*1)
    ds_y = np.mean((np.diff(np.sign(slopez_y), axis=0) != 0)*1)

    # La mitad peaks y la otra mitad pits; la longitud de onda media será:
    l_x = STEP / (ds_x/2)
    l_y = STEP / (ds_y/2)

    # Quitamos el apellido al filename (las hojas excel aceptan hasta 31 char.)
    filename = filename.replace('.xyz', '')

    if USE_DETREND:
        filename = filename + "UT"
    if FILTER_Z:
        filename = filename + "Z"

    # Diccionario con los resultados que queremos guardar
    results = {
        'fichero': filename,
        'filter' : FILTER_Z,
        'step' : STEP,
        'proporción' : PROP,
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

    # Se pason los resultados a un dataframe y se guardan en el excel
    pdsave = pd.DataFrame.from_dict(results, orient='index')
    pdsave.to_excel(writer, sheet_name=filename)

    # Definición de algunas variables utilizadas en gráficos
    ap = aspectrad[~aspectrad.mask].ravel()
    aspectgrad = 90 - ap * 180 / np.pi
    aspect_hist, aspect_hist_bin = np.histogram(aspectgrad, bins=256)
    frecang = aspect_hist/np.sum(aspect_hist)
    ang = aspect_hist_bin*np.pi/180
    angp = np.mean(ang)*180/np.pi
    anf = np.diff(ang)

    if MAKE_PLOTS:

        # Crear esta figura consume mucho tiempo
        if PLOT_RAW:
            fig_raw, ax_raw = plt.subplots()
            ax_raw.axis('equal')
            ax_raw.set_xlabel(r'$X  (\mu m)$')
            ax_raw.set_ylabel(r'$Y  (\mu m)$')
            ax_raw.set_title('raw data')
            pos_raw = ax_raw.scatter(xo, yo, c=zo, vmin=zmino, vmax=zmaxo)
            bar_raw = fig_raw.colorbar(pos_raw)
            bar_raw.ax.set_ylabel(r'$Z  (\mu m)$', labelpad=0, y=0.5)
            fig_nam = 'raw data' + filename
            fig_raw.savefig(fig_nam, dpi=300)

        # Z's interpoladas
        titt = 'raw data_int'
        bar_lab = r'$Z  (\mu m)$'
        fig_grid(grid_z1, titt, bar_lab, extension, filename)

        # Z's gradient:  x and y (also with slope)
        titt = 'gradient x data'
        bar_lab = r'$Z grad_x$'
        fig_grid(gradz_x, titt, bar_lab, extension, filename)

        titt = 'slope x'
        fig_grid(slopez_x, titt, bar_lab, extension, filename)

        titt = 'gradient y data'
        bar_lab = r'$Z grad_y$'
        fig_grid(gradz_y, titt, bar_lab, extension, filename)

        titt = 'slope y'
        bar_lab = r'$Z grad_y$'
        fig_grid(slopez_y, titt, bar_lab, extension, filename)

        # Se traza un hill shade con los datos"
        titt = 'raw data_hs'
        fig_grid2D(grid_z1, titt, extension, filename)

        # z hist
        xlab = r'$z$'
        ylab = r'$Data_number$'
        titt = 'z distribution density'
        fig_hist(z, xlab, ylab, titt, filename)

        # Grad_x Slope hist
        xlab = r'$grad_z$'
        titt = 'x Slope hist'
        fig_hist(ravez_x, xlab, ylab, titt, filename)

        # Grad_y Slope hist
        titt = 'y Slope hist'
        fig_hist(ravez_y, xlab, ylab, titt, filename)

        # Aspect angle hist
        xlab = r'$Direction (ºN)$'
        titt = 'Aspect angle hist'
        fig_hist(aspectgrad, xlab, ylab, titt, filename)

        # Polar diagram
        filename_ros = filename +"_roseta.png"
        fig_roseta(ang, frecang, anf, filename_ros)

        # Z`s trends in x and y direction
        fig_trend, ax_trend = plt.subplots()
        ax_trend.plot(grid_y[:, 0], meanZ_y, 'r')
        ax_trend.plot(grid_x[0, :], meanZ_x)
        ax_trend.set_xlabel('X (blue) or Y (red)')
        ax_trend.set_ylabel('z')
        ax_trend.set_title('x/y trends')
        fig_nam = 'x and y trends' + filename
        fig_trend.savefig(fig_nam, dpi=300)

        # Se cierran las figuras (con plt.ioff no es necesario pero...)
        plt.close('all')

# Se guardan los resultados en disco y se cierra el fichero
if USE_WRITER:
    writer.save()
    writer.close()

# Utilidades
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
                     antialiased=True, cmap=cm.gist_earth)
    ax1.set_xlabel(r'$X  (\mu m)$')
    ax1.set_ylabel(r'$Y  (\mu m)$')
    ax1.set_zlabel(r'$Z  (\mu m)$')

if USE_MAYAVI:
    from mayavi import mlab
    # Por ejemplo:
    mlab.surf(grid_x, grid_y, grid_z1)
