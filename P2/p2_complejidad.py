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
from sklearn.metrics import accuracy_score, balanced_accuracy_score

#
# PARÁMETROS GLOBALES
#

SEED = 2020
IMG_PATH = "img/"
SAVE_FIGURES = False

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior."""

    input("\n(Pulsa [Enter] para continuar...)\n")
    plt.close()

def scatter_plot(X, axis, y = None, fun = None, label = None,
                 title = None, regions = False, figname = ""):
    """Muestra un scatter plot con leyenda y (opcionalmente) la frontera
       de un cuadrático con dos grados de libertad, de la forma
       w0 + w1 * x1 + w2 * x2 + w3 * x1 * x2 + w4 * x1^2 + w5 * x2^2.
         - X: matriz de características bidimensional.
         - axis: nombres de los ejes.
         - y: vector de etiquetas o clases.
         - fun: función que toma dos parámetros tal que f(x, y) = 0 define
           la frontera del clasificador.
         - label: etiqueta del clasificador.
         - title: título del plot.
         - regions: controla si se pintan las regiones en las que el clasificador
           divide al plano.
         - figname: nombre para guardar la gráfica en fichero."""

    # Establecemos tamaño, colores e información del plot
    fig = plt.figure(figsize = (8, 6))
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    cmap = ListedColormap(['r', 'lime'])
    if title is not None:
        plt.title(title)

    # Establecemos los límites del plot
    xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
    ymin, ymax = np.min(X[:, 1]), np.max(X[:, 1])
    scale_x = (xmax - xmin) * 0.01
    scale_y = (ymax - ymin) * 0.01
    plt.xlim(xmin - scale_x, xmax + scale_x)
    plt.ylim(ymin - scale_y, ymax + scale_y)

    # Asignamos vector de etiquetas
    if y is None:
        c = [1 for _ in range(len(X))]
    else:
        c = y

    # Mostramos scatter plot con leyenda
    scatter = plt.scatter(X[:, 0], X[:, 1],
        cmap = cmap, c = c, edgecolors = 'k')
    if y is not None:
        legend1 = plt.legend(
            *scatter.legend_elements(),
            title = "Clases",
            loc = "upper right")

    # Pintamos el clasificadores
    if fun is not None:
        xx, yy = np.meshgrid(np.linspace(xmin - scale_x, xmax + scale_x, 100),
            np.linspace(ymin - scale_y, ymax + scale_y, 100))

        # Función que define el clasificador
        z = fun(xx, yy)

        # Pintamos las regiones en las que el clasificador divide al plano
        if regions:
            cont = plt.contourf(xx, yy, z,
                cmap = cmap, levels = 0, alpha = 0.1)
            fig.colorbar(cont, aspect = 30, label = "f(x, y)")

        # Pintamos el clasificador como la curva de nivel 0 en el plano de
        # la función 'fun'
        plt.contour(xx, yy, z,
            levels = [0], colors = ['tab:blue'],
            linewidths = 2).collections[0].set_label(label)

        # Añadimos la leyenda del clasificador
        if label is not None:
            plt.legend(loc = "lower right")

    # Añadimos leyenda sobre las clases
    if y is not None:
        plt.gca().add_artist(legend1)

    if SAVE_FIGURES:
        plt.savefig(IMG_PATH + figname + ".png")
    else:
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

    # Generamos las nubes de puntos
    X_unif = uniform_sample(50, 2, -50, 50)
    X_gauss = gaussian_sample(50, 2, [5, 7])

    # Mostramos los resultados
    print("--- Apartado a)")
    scatter_plot(X_unif, ["x", "y"],
        title = "Nube de 50 puntos uniformes",
        figname = "ex1-1-1")
    print("--- Apartado b)")
    scatter_plot(X_gauss, ["x", "y"],
        title = "Nube de 50 puntos gaussianos",
        figname = "ex1-1-2")

#
# EJERCICIO 2: SEPARACIÓN LINEAL CON RUIDO
#

def sign(x):
    """Devuelve el signo de x, considerando que el signo de 0 es 1."""

    return 1 if x >= 0 else -1

def f(a, b, x, y):
    """Función utilizada para asignar etiquetas binarias a los puntos.
       Mide el signo de la distancia del punto (x, y) a la recta de pendiente
       'a' y término independiente 'b'. Se considera que los puntos estén
       sobre la recta pertenecen a la clase del 1."""

    return sign(y - a * x - b)

def simulate_noise(y, p):
    """Introduce ruido en el vector de etiquetas 'y', cambiando una
       fracción 'p' de las positivas y la misma fracción de las negativas."""

    y_noise = np.copy(y)
    for label in [-1, 1]:
        # Seleccionamos los índices de las clases positivas (resp. negativas)
        idxs = np.where(y == label)[0]

        # Elegimos aleatoriamente una fracción 'p' de ellos
        random_idxs = np.random.choice(idxs, int(p * len(idxs)), replace = False)

        # Cambiamos el signo de los elegidos
        y_noise[random_idxs] = -y_noise[random_idxs]

    return y_noise

def ex2():
    """Dibuja puntos de forma uniforme, etiquetados según el signo de su
       distancia a una recta también elegida uniformemente, con y sin ruido.
       Posteriormente compara la clasificación realizada por la recta y por
       distintos clasificadores cuadráticos."""

    # Volvemos a establecer la semilla para poder obtener posteriormente
    # los mismos puntos con las mismas etiquetas
    np.random.seed(SEED)

    # Generamos 100 puntos y una recta en [-50, 50] x [-50, 50]
    N = 100
    X = uniform_sample(N, 2, -50, 50)
    a, b = line_sample(-50, 50)
    v = lambda x, y: y - a * x - b

    # Etiquetamos los puntos según el signo de la distancia a la recta
    y = np.array([f(a, b, x[0], x[1]) for x in X])

    # Mostramos el resultado de la clasificación
    scatter_plot(X, ["x", "y"], y,
        v, "Recta y = {:0.3f}x + ({:0.3f})".format(a, b),
        title = "Clasificación dada por una recta (sin ruido)",
        figname = "ex1-2-1")

    # Introducimos ruido en las etiquetas
    y_noise = simulate_noise(y, 0.1)

    # Mostramos el resultado de la nueva clasificación
    scatter_plot(X, ["x", "y"], y_noise,
        v, "Recta y = {:0.3f}x + ({:0.3f})".format(a, b),
        title = "Clasificación dada por una recta (con ruido)",
        figname = "ex1-2-2")

    # Listamos los clasificadores para comparar
    classifiers = [
        v,
        lambda x, y: (x - 10) ** 2 + (y - 20) ** 2 - 400,
        lambda x, y: 0.5 * (x + 10) ** 2 + (y - 20) ** 2 - 400,
        lambda x, y: 0.5 * (x - 10) ** 2 - (y + 20) ** 2 - 400,
        lambda x, y: y - 20 * x ** 2 - 5 * x + 3]

    classifiers_names = [
        ("f(x,y) = y - ({:0.3f})x - ({:0.3f}) = 0".format(a, b), "recta"),
        (r"$f(x, y) = (x-10)^2 + (y-20)^2 - 400 = 0$", "elipse 1"),
        (r"$f(x, y) = 0.5(x+10)^2 + (y-20)^2 - 400 = 0$", "elipse 2"),
        (r"$f(x, y) = 0.5(x-10)^2 - (y+20)^2 - 400 = 0$", "hipérbola"),
        (r"$f(x, y) = y - 20x^2 - 5x + 3 = 0$", "parábola")]

    # Mostramos las gráficas
    for fun, (label, name) in zip(classifiers, classifiers_names):
        scatter_plot(X, ["x", "y"], y_noise, fun, label,
            title = "Frontera de clasificación para la " + name,
            regions = True,
            figname = "ex1-2-" + name)

    # Mostramos la proporción de clases
    positive_prop = np.sum(y_noise == 1) / len(y_noise)
    print("Proporción de clases:")
    print("    Clase del 1: {:0.3f}%".format(100 * positive_prop))
    print("    Clase del -1: {:0.3f}%\n".format(100 * (1 - positive_prop)))

    # Mostramos las métricas de clasificación
    for fun, (_, name) in zip(classifiers, classifiers_names):
        y_pred = [sign(fun(x[0], x[1])) for x in X]
        acc = accuracy_score(y_noise, y_pred, normalize = True)
        balanced_acc = balanced_accuracy_score(y_noise, y_pred)

        print("Clasificador: " + name)
        print("    Accuracy = {:0.3f}%".format(acc * 100))
        print("    Balanced accuracy = {:0.3f}%\n".format(balanced_acc * 100))

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Función principal. Ejecuta el ejercicio paso a paso."""

    # Semilla aleatoria para reproducibilidad
    np.random.seed(SEED)

    # Número de decimales fijo para salida de vectores
    np.set_printoptions(formatter = {'float': lambda x: "{:0.3f}".format(x)})

    print("-------- EJERCICIO SOBRE COMPLEJIDAD Y RUIDO --------\n")
    print("--- EJERCICIO 1 ---")
    ex1()
    print("--- EJERCICIO 2 ---")
    ex2()

if __name__ == "__main__":
    main()
