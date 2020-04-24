#!/usr/bin/env python
# coding: utf-8
# uso: ./p2_digitos.py

##########################################################################
# Aprendizaje Automático. Curso 2019/20.
# Práctica 2: Modelos lineales.
# Ejercicio sobre clasificación de dígitos (bonus)
# Antonio Coín Castro. Grupo 3.
##########################################################################

#
# LIBRERÍAS
#

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

#
# PARÁMETROS GLOBALES
#

PATH = "datos/"
SEED = 2020
EPS = 1e-10
CLASSES = [4, 8]

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior."""

    input("\n(Pulsa [Enter] para continuar...)\n")
    plt.close()

def scatter_plot(X, axis, y, ws, labels, title = None):
    """Muestra un scatter plot de puntos etiquetados y varias rectas.
         - X: matriz de características de la forma [1, x1, x2].
         - axis: nombres de los ejes.
         - y: vector de etiquetas o clases.
         - ws: lista de vectores 3-dimensionales que representan cada recta.
         - labels: lista de etiquetas de las rectas.
         - title: título del plot."""

    # Establecemos tamaño, colores e información del plot
    plt.figure(figsize = (8, 6))
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    cmap = ListedColormap(['mediumpurple', 'gold'])
    if title is not None:
        plt.title(title)

    # Establecemos los límites del plot
    xmin, xmax = np.min(X[:, 1]), np.max(X[:, 1])
    ymin, ymax = np.min(X[:, 2]), np.max(X[:, 2])
    scale_x = (xmax - xmin) * 0.01
    scale_y = (ymax - ymin) * 0.01
    plt.xlim(xmin - scale_x, xmax + scale_x)
    plt.ylim(ymin - scale_y, ymax + scale_y)

    # Mostramos scatter plot con leyenda
    scatter = plt.scatter(X[:, 1], X[:, 2],
        cmap = cmap, c = y, edgecolors = 'k')
    if y is not None:
        legend1 = plt.legend(
            *scatter.legend_elements(),
            title = "Clases",
            loc = "upper right")

    # Pintamos la recta con leyenda
    if ws is not None:
        for w, l in zip(ws, labels):
            x = np.array([xmin - scale_x, xmax + scale_x])
            plt.plot(x, (-w[0] - w[1] * x) / w[2], label = l, linewidth = 2)

        plt.legend(loc = "lower right")

    # Añadimos leyenda sobre las clases
    if y is not None:
        plt.gca().add_artist(legend1)

    plt.show(block = False)
    wait()

def read_data(file_X, file_y):
    """Leer archivos de características y clases en un par de vectores.
       El formato de salida de las características es [1, x1, x2].
         - file_X: archivo con las características bidimensionales.
         - file_y: archivo con las clases."""

    # Leemos los ficheros
    data_X = np.load(file_X)
    data_y = np.load(file_y)
    X = []
    y = []

    # Solo guardamos los datos de dos clases
    for i in range(0, data_y.size):
        if data_y[i] in CLASSES:
            if data_y[i] == CLASSES[0]:
                y.append(-1)
            else:
                y.append(1)
            X.append(np.array([1, data_X[i][0], data_X[i][1]]))

    X = np.array(X, np.float64)
    y = np.array(y, np.float64)

    return X, y

def sign(x):
    """Devuelve el signo de 'x', considerando que es 1 si x = 0."""

    return 1 if x >= 0 else -1

#
# IMPLEMENTACIÓN DEL BONUS
#

def err(X, y, w):
    """Devuelve el error de clasificación del hiperplano dado por 'w' para un
       conjunto de datos  homogéneos 'X' con verdaderas etiquetas 'y'."""
    
    incorrect = [sign(x.dot(w)) for x in X] != y
    return np.mean(incorrect)

def pseudoinverse(X, y):
    """Obtiene un vector de pesos a través del método de la pseudoinversa
       para resolver el problema de mínimos cuadrados. Se emplea la
       descomposición SVD para evitar el cálculo de matrices inversas.
         - X: matriz de características con primera componente 1.
         - y: vector de etiquetas."""

    dims = len(X[0])

    # Realizamos la descomposición SVD y calculamos la solución
    u, s, vt = np.linalg.svd(X)
    d = np.diag([1 / l if l > EPS else 0.0 for l in s])
    return (vt.T @ d @ u.T[0:dims]) @ y

def pla_pocket(X, y, max_it, w_ini):
    """Ajusta los parámetros de un hiperplano para un problema de clasificación
       binaria usando el algoritmo PLA-Pocket.
         - X: matriz de datos en coordenadas homogéneas (primera componente 1).
         - y: vector de etiquetas (1 ó -1).
         - max_it: número fijo de iteraciones.
         - w_ini: vector de pesos inicial."""

    w = w_ini.copy()
    w_best = w.copy()
    best_err = err(X, y, w_best)

    for _ in range(max_it):
        for x, l in zip(X, y):
            if sign(x.dot(w)) != l:
                w += l * x

        curr_err = err(X, y, w)
        if curr_err < best_err:
            best_err = curr_err
            w_best = w.copy()

    return w_best

def err_bound_hoeffding(err, n, m, delta):
    """Cota de Hoeffding para el error de generalización.
         - err: error a partir del cual generalizamos.
         - n: tamaño del conjunto usado para calcular 'err'.
         - m: tamaño de la clase de hipótesis usada para calcular 'err'.
         - delta: tolerancia."""

    return err + np.sqrt((1 / (2 * n)) * np.log((2 * m) / delta))

def err_bound_vc(err, n, vc, delta):
    """Cota para el error de generalización basada en la dimensión VC.
         - err: error a partir del cual generalizamos.
         - n: tamaño del conjunto usado para calcular 'err'.
         - vc: dimensión VC del clasificador usado para calcular 'err'.
         - delta: tolerancia."""

    return err + np.sqrt((8 / n) * np.log(4 * ((2 * n) ** vc + 1) / delta))

def bonus():
    """Función que implementa el bonus. Consiste en abordar un problema
       de clasificación de dígitos mediante un modelo de regresión
       lineal y mediante el algoritmo PLA-Pocker."""

    # Cargamos los datos
    X_train, y_train = read_data(PATH + "X_train.npy", PATH + "y_train.npy")
    X_test, y_test = read_data(PATH + "X_test.npy", PATH + "y_test.npy")

    # Estimamos un modelo con con pseudoinversa y otro con PLA-Pocket
    max_it = 1000
    w_pseudo = pseudoinverse(X_train, y_train)
    w_pocket_zeros = pla_pocket(X_train, y_train, max_it, np.random.rand(3))
    w_pocket_pseudo = pla_pocket(X_train, y_train, max_it, w_pseudo)

    ws = [w_pseudo, w_pocket_zeros, w_pocket_pseudo]
    names = ["Pseudoinversa", "PLA-Pocket (aleatorio)", "PLA-Pocket (pseudoinversa)"]

    # Mostramos los resultados
    for w, name in zip(ws, names):
        n_in = len(X_train)
        n_test = len(X_test)
        delta = 0.05
        e_in = err(X_train, y_train, w)
        e_test = err(X_test, y_test, w)

        print("\n---- {}".format(name))
        print("Vector de pesos =", w)
        print("Errores:")
        print("    E_in = {:0.5f}".format(e_in))
        print("    E_test = {:0.5f}".format(e_test))
        print("Cotas para E_out:")
        print("    Cota usando E_in (VC) = {:0.5f}"
            .format(err_bound_vc(e_in, n_in, 3, delta)))
        print("    Cota usando E_in (Hoeffding) = {:0.5f}"
            .format(err_bound_hoeffding(e_in, n_in, 2 ** (64 * 3), delta)))
        print("    Cota usando E_test (Hoeffding) = {:0.5f}"
            .format(err_bound_hoeffding(e_test, n_test, 1, delta)))

    # Mostramos los conjuntos de training y test junto con las rectas obtenidas
    scatter_plot(X_train, ["Intensidad promedio", "Simetría"],
        y_train, ws, names,
        title = "Conjunto de entrenamiento junto a las rectas estimadas")
    scatter_plot(X_test, ["Intensidad promedio", "Simetría"],
        y_test, ws, names,
        title = "Conjunto de test junto a las rectas estimadas")

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Función principal. Ejecuta el ejercicio paso a paso."""

    # Semilla aleatoria para reproducibilidad
    np.random.seed(SEED)

    # Número de decimales fijo para salida de vectores
    np.set_printoptions(formatter = {'float': lambda x: "{:0.3f}".format(x)})

    print("-------- EJERCICIO DE BONUS: CLASIFICACIÓN DE DÍGITOS --------")
    bonus()

if __name__ == "__main__":
    main()
