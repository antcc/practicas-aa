#!/usr/bin/env python
# coding: utf-8
# uso: ./p2_complejidad.py

##########################################################################
# Aprendizaje Automático. Curso 2019/20.
# Práctica 2: Modelos lineales.
# Ejercicio sobre complejidad y ruido.
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
SEED = 2021

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior."""

    input("\n(Pulsa [Enter] para continuar...)\n")
    plt.close()

def scatter_plot(X, axis, y = None, ws = None, ws_labels = None, title = None):
    """Muestra un scatter plot con leyenda y (opcionalmente) rectas
       de regresión.
         - X: vector de características bidimensional.
         - axis: nombres de los ejes.
         - y: vector de etiquetas o clases.
         - ws: vectores de pesos
         - ws_labels: etiquetas de los vectores de pesos. Debe
           aparecer obligatoriamente si 'ws' no es vacío.
         - title: título del plot."""

    # Establecemos tamaño e información del plot
    plt.figure(figsize = (8, 6))
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    if title is not None:
        plt.title(title)

    # Establecemos los límites del plot
    xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
    ymin, ymax = np.min(X[:, 1]), np.max(X[:, 1])
    plt.xlim(xmin - 1, xmax + 1)
    plt.ylim(ymin - 1, ymax + 1)

    # Asignamos vector de etiquetas
    if y is None:
        c = [1 for _ in range(len(X))]
    else:
        c = y

    # Mostramos scatter plot con leyenda
    scatter = plt.scatter(X[:, 0], X[:, 1], c = c,
        cmap = ListedColormap(['r', 'limegreen']), edgecolors = 'k')
    if y is not None:
        legend1 = plt.legend(
            *scatter.legend_elements(),
            title = "Clases",
            loc = "upper right")

    # Pintamos las rectas
    if ws is not None:
        x = np.array([xmin, xmax])
        for w, l in zip(ws, ws_labels):
            plt.plot(x, (-w[0] - w[1] * x) / w[2], label = l, linewidth = 2)

        plt.legend(loc = "lower right")

    # Añadimos leyenda sobre las clases
    if y is not None:
        plt.gca().add_artist(legend1)

    plt.show(block = False)
    wait()

def uniform_sample(n, d, low, high):
    """Genera 'n' puntos de dimension 'd' uniformente distribuidos en el hipercubo
       definido por [low, high]."""

    return np.random.uniform(low, high, (n, d))

def gaussian_sample(n, d, sigma, mean = 0.0):
    """Genera 'n' puntos de dimensión 'd' distribuidos según una
       gaussiana multivariante con vector de medias 'mean' y matriz
       de covarianzas diagonal definida por el vector 'sigma'."""

    return np.random.normal(mean, np.sqrt(sigma), (n, d))

def line_sample(low, high):
    """Simula de manera uniforme los parámetros de una recta
       y = ax + b que corte al cuadrado [low, high] x [low, high]."""

    # Generamos uniformemente dos puntos en el cuadrado
    points = np.random.uniform(low, high, size = (2, 2))
    x1, x2 = points[:, 0]
    y1, y2 = points[:, 1]

    # Calculamos la pendiente y el término independiente
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    return a, b

#
# EJERCICIO 1: NUBE DE PUNTOS ALEATORIOS
#

def ex1():
    """Visualización de nubes de puntos aleatorios."""

    X_unif = uniform_sample(50, 2, -50, 50)
    X_gauss = gaussian_sample(50, 2, [5, 7])

    print("  --- Apartado a)")
    scatter_plot(X_unif, ["x", "y"],
        title = "Nube de puntos uniformes")
    print("  --- Apartado b)")
    scatter_plot(X_gauss, ["x", "y"],
        title = "Nube de puntos gaussianos")

#
# EJERCICIO 2: SEPARACIÓN LINEAL CON RUIDO
#

def f(a, b, x, y):
    """Función utilizada para asignar etiquetas binarias a
       los puntos. Mide el signo de la distancia del punto
       (x, y) a la recta y = ax + b. Se considera que los
       puntos estén sobre la recta pertenecen a la clase del 1."""

    return 1 if y - a * x - b >= 0 else -1

def simulate_noise(y, p):
    """Introduce ruido en el vector de etiquetas 'y', cambiando una
       fracción 'p' de las positivas y la misma fracción de las negativas."""

    y_noise = np.copy(y)
    for label in [-1, 1]:
        idxs = np.where(y == label)[0]
        random_idxs = np.random.choice(idxs, int(p * len(idxs)), replace = False)
        y_noise[random_idxs] = -y_noise[random_idxs]

    return y_noise

def ex2():
    """Dibuja puntos etiquetados según el signo de su distancia a
       una recta, con y sin ruido."""

    # Generamos 100 puntos y una recta en [-50, 50] x [-50, 50]
    X = uniform_sample(100, 2, -50, 50)
    a, b = line_sample(-50, 50)

    # Etiquetamos los puntos según el signo de la distancia a la recta
    y = np.array([f(a, b, x[0], x[1]) for x in X])

    # Mostramos el resultado de la clasificación
    scatter_plot(X, ["x", "y"], y,
        [[-b, -a, 1]], ["Recta y = {:0.5f}x + {:0.5f}".format(a, b)],
        title = "Clasificación dada por una recta (sin ruido)")

    # Introducimos ruido en las etiquetas
    y_noise = simulate_noise(y, 0.1)

    # Mostramos el resultado de la nueva clasificación
    scatter_plot(X, ["x", "y"], y_noise,
        [[-b, -a, 1]], ["Recta y = {:0.5f}x + {:0.5f}".format(a, b)],
        title = "Clasificación dada por una recta (con ruido)")

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Función principal. Ejecuta el ejercicio paso a paso."""

    # Semilla aleatoria para reproducibilidad
    np.random.seed(SEED)

    # Número de decimales fijo para salida de vectores
    np.set_printoptions(formatter = {'float': lambda x: "{:0.5f}".format(x)})

    print("-------- EJERCICIO SOBRE COMPLEJIDAD Y RUIDO --------")
    print("--- EJERCICIO 1 ---")
    #ex1()
    print("\n--- EJERCICIO 2 ---")
    ex2()
    """wait()
    print("\n--- EJERCICIO 3 ---")
    ex3()"""

if __name__ == "__main__":
    main()
