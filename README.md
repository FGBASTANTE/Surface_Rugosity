# Surface_Rugosity
Script en Python que lee ficheros de puntos (X, Y, Z) no estructurados y, para cada uno de ellos, calcula una serie de parámetros indicadores de la rugosidad de la superficie z = z(x, y).
Los ficheros de puntos son ficheros de texto estándar con tres columnas con las coordenadas originales de los puntos xo, yo y zo.
Permite eliminar la tendencia de la superficie con un polinomio en X e Y del grado deseado.
También permite elegir para el análisis una zona central reducida de la superficie, o eliminar valores de Z anómalos (outlers) en base a su desviación estándar.
El script crea un grid e interpola el valor de las Z`s en el mismo a partir de los datos originales (o de los datos una vez eliminada la tendencia).
Los parámetros calculados se guardan en un fichero excel (en memoria también están como un diccionario y como panda dataframe).
También crea y guarda multitud de gráficos relativos a los parámetros si se desea.

Mis agradecimientos a la gran comunidad Python que comparte de forma altruista su trabajo y conocimiento.

@author: Fernando García Bastante
Universidad de Vigo
