#!/usr/bin/env python
# coding: utf-8
# uso: ./p0.py

##########################################################################
# Aprendizaje Automático. Curso 2019/20.
# Práctica 0: Introducción a Python.
# Antonio Coín Castro. Grupo 3.
##########################################################################

#
# LIBRERÍAS
#

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

#
# PARÁMETROS GLOBALES
#

#
# EJERCICIO 1:
#

def ex1():
    """Realiza una serie de acciones con scikit-learn:
         1. Leer la base de datos IRIS.
         2. Obtener las características (X) y las clases (y).
         3. Quedarse con las dos últimas características,
         4. Mostrar en un scatter plot los datos, coloreando cada clase
            con un color distinto, con una leyenda."""

    # Importamos la base de datos IRIS
    iris = datasets.load_iris()

    # Obtenemos todas las características y las clases
    X = iris.data
    y = iris.target

    # Mostramos los nombres de las características y las clases
    feature_names = iris.feature_names
    class_names = iris.target_names
    print("Características:", feature_names)
    print("Etiquetas:", class_names)

    # Nos quedamos con las dos últimas características
    X = X[:, -2:]

    # Mostramos la información en un scatter plot con leyenda
    plt.figure(figsize = (8, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c = y, cmap = ListedColormap(['r', 'g', 'b']), edgecolors = 'k')
    plt.legend(handles = scatter.legend_elements()[0], title = "Classes", labels = class_names.tolist(), loc = "upper left")
    plt.xlabel(feature_names[-2])
    plt.ylabel(feature_names[-1])
    plt.show()

#
# FUNCIÓN PRINCIPAL
#

def main():
    print("--- EJERCICIO 1 ---")
    ex1()

if __name__ == "__main__":
    main()
