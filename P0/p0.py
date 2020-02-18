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
from sklearn.model_selection import train_test_split

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior"""

    input("(Pulsa [Enter] para continuar...)")
    plt.close()

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
    plt.figure(figsize = (8, 6))
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c = y, cmap = ListedColormap(['r', 'g', 'b']), edgecolors = 'k')
    plt.legend(
        handles = scatter.legend_elements()[0], title = "Classes",
        labels = class_names.tolist(), loc = "upper left")
    plt.xlabel(feature_names[-2])
    plt.ylabel(feature_names[-1])
    plt.show(block = False)
    wait()

def ex2():
    """Separa los datos de IRIS en training (80%) y test (20%),
       respetando la proporción de clases."""

    # Cargamos los datos
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    n = len(iris.target_names)

    # Realizamos el split de train-test.
    # Con el parámetro stratify = y mantenemos la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size = 0.8, shuffle = True, stratify = y)

    # Mostramos los resultados
    print("Número de ejemplos de entrenamiento:", X_train.shape[0])
    print("Primeros 5 ejemplos de entrenamiento y sus clases:")
    for pair in list(zip(X_train[:5].tolist(), y_train[:5])):
        print(pair)

    print("Número de ejemplos de test:", X_test.shape[0])
    print("Primeros 5 ejemplos de test y sus clases:")
    for pair in list(zip(X_test[:5].tolist(), y_test[:5])):
        print(pair)

    # Comprobamos que se mantiene la proporción de clases
    for i in range(n):
        orig_prop = len(y[y == i]) / y.shape[0]
        train_prop = len(y_train[y_train == i]) / y_train.shape[0]
        test_prop = len(y_test[y_test == i]) / y_test.shape[0]

        print("Proporción de la clase", i)
        print("Dataset completo:", orig_prop)
        print("Training:", train_prop)
        print("Test:", test_prop)
    wait()

def ex3():
    """Representa algunas funciones trigonométricas en [0, 2π]."""

    # Obtenemos 100 valores equiespaciados en [0, 2π]
    x = np.linspace(0, 2 * np.pi, 100)

    # Calculamos el valor de las funciones en esos puntos
    f1 = np.sin(x)
    f2 = np.cos(x)

    # Representamos las tres funciones en la misma gráfica
    plt.plot(x, f1, 'k', linestyle = "--", label = "sin(x)")
    plt.plot(x, f2, 'b', linestyle = "--", label = "cos(x)")
    plt.plot(x, f1 + f2, 'r', linestyle = "--", label = "sin(x) + cos(x)")
    plt.legend()
    plt.show(block = False)
    wait()


#
# FUNCIÓN PRINCIPAL
#

def main():
    print("--- EJERCICIO 1 ---")
    ex1()
    print("\n--- EJERCICIO 2 ---")
    ex2()
    print("\n--- EJERCICIO 3 ---")
    ex3()

if __name__ == "__main__":
    main()
