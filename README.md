# Surface_Rugosity

Python script that reads unstructured point files (X, Y, Z) and, for each of them, computes a set of parameters indicating the surface roughness z = z(x, y). The point files are standard text files with three columns containing the original coordinates of the points xo, yo and zo. It allows to remove the surface trend with a polynomial in X and Y of the desired degree. It also allows to choose for the analysis a reduced central zone of the surface, or to eliminate outliers based on their standard deviation. The script creates a grid and interpolates the value of the Z`s in it from the original data (or from the data after trend removal). The calculated parameters are saved in an excel file (in memory they are also as a dictionary and as a panda dataframe). It also creates and saves a multitude of plots relative to the parameters if desired.

Pérez-Rey, I., Bastante, F. G., Alejano, L. R., & Ivars, D. M. (2020, August). Influence of Microroughness on the Frictional Behavior and Wear Response of Planar Saw-Cut Rock Surfaces. International Journal of Geomechanics. American Society of Civil Engineers (ASCE). http://doi.org/10.1061/(asce)gm.1943-5622.0001742

...........................................................................................................................................................................................\
\
Script en Python que lee ficheros de puntos (X, Y, Z) no estructurados y, para cada uno de ellos, calcula una serie de parámetros indicadores de la rugosidad de la superficie z = z(x, y).
Los ficheros de puntos son ficheros de texto estándar con tres columnas con las coordenadas originales de los puntos xo, yo y zo.
Permite eliminar la tendencia de la superficie con un polinomio en X e Y del grado deseado.
También permite elegir para el análisis una zona central reducida de la superficie, o eliminar valores de Z anómalos (outliers) en base a su desviación estándar.
El script crea un grid e interpola el valor de las Z`s en el mismo a partir de los datos originales (o de los datos una vez eliminada la tendencia).
Los parámetros calculados se guardan en un fichero excel (en memoria también están como un diccionario y como panda dataframe).
También crea y guarda multitud de gráficos relativos a los parámetros si se desea.

Mis agradecimientos a la gran comunidad Python que comparte de forma altruista su trabajo y conocimiento.

@author: Fernando García Bastante
Universidad de Vigo
