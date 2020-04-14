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

SEED = 2020

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior."""

    input("\n(Pulsa [Enter] para continuar...)\n")
    plt.close()

def scatter_plot(X, axis, y = None, ws = None,
                 ws_labels = None, title = None, ncols = 2):
    """Muestra un scatter plot con leyenda y (opcionalmente) la frontera
       de varios clasificadores cuadráticos, de la forma
       w0 + w1 * x1 + w2 * x2 + w3 * x1 * x2 + w4 * x1^2 + w5 * x2^2.
         - X: vector de características bidimensional.
         - axis: nombres de los ejes.
         - y: vector de etiquetas o clases.
         - ws: vectores de pesos de los clasificadores (6-dimensionales).
         - ws_labels: etiquetas de los vectores de pesos. Debe
           aparecer obligatoriamente si 'ws' no es vacío.
         - title: título del plot.
         - ncols: número de columnas para el subplot."""

    # Establecemos tamaño e información del plot
    plt.figure(figsize = (8, 6))
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
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
    scatter = plt.scatter(X[:, 0], X[:, 1], c = c, edgecolors = 'k')
    if y is not None:
        legend1 = plt.legend(
            *scatter.legend_elements(),
            title = "Clases",
            loc = "upper right")

    # Pintamos los clasificadores
    if ws is not None:
        xx, yy = np.meshgrid(np.linspace(xmin - scale_x, xmax + scale_x, 100),
            np.linspace(ymin - scale_y, ymax + scale_y, 100))

        # Función que encapsula el producto escalar X * w
        h = lambda x, y, w: np.array([1, x, y, x * y, x * x, y * y]).dot(w)

        # Iteramos sobre los clasificadores (cada uno de un color)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(ws)))
        for w, l, c in zip(ws, ws_labels, colors):
            # Pintamos la curva de nivel 0 en el plano de la función z = X * w
            z = h(xx, yy, w)
            plt.contourf(xx, yy, z, levels = 0,  alpha = 0.2)
            plt.contour(xx, yy, z, levels = [0],
                linewidths = 2).collections[0].set_label(l)
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

    # Generamos las nubes de puntos
    X_unif = uniform_sample(50, 2, -50, 50)
    X_gauss = gaussian_sample(50, 2, [5, 7])

    # Mostramos los resultados
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
        # Seleccionamos los índices de las clases positivas (resp. negativas)
        idxs = np.where(y == label)[0]

        # Elegimos aleatoriamente un 10% de ellos
        random_idxs = np.random.choice(idxs, int(p * len(idxs)), replace = False)

        # Cambiamos el signo de los elegidos
        y_noise[random_idxs] = -y_noise[random_idxs]

    return y_noise

def random_line_classifier(noise = False, show = False):
    """Genera una nube de puntos de forma uniforme y una recta
       a partir de la cual clasificarlos, con eventual ruido.
       Devuelve los puntos, la recta y las etiquetas generadas.
         - noise: controla si se introduce ruido en las etiquetas.
         - show: controla si se muestran gráficas."""

    # Generamos 100 puntos y una recta en [-50, 50] x [-50, 50]
    X = uniform_sample(100, 2, -50, 50)
    a, b = line_sample(-50, 50)

    # Etiquetamos los puntos según el signo de la distancia a la recta
    y = np.array([f(a, b, x[0], x[1]) for x in X])

    # Introducimos ruido en las etiquetas
    if noise:
        y = simulate_noise(y, 0.1)

    # Mostramos el resultado de la clasificación
    if show:
        scatter_plot(X, ["x", "y"], y,
            [[-b, -a, 1, 0, 0, 0]], ["Recta y = {:0.5f}x + ({:0.5f})".format(a, b)],
            title = "Clasificación dada por una recta (sin ruido)")

    return X, y, a, b

def ex2():
    """Dibuja puntos etiquetados según el signo de su distancia a
       una recta, con y sin ruido."""

    # Clasificación sin ruido
    X, y, a, b = random_line_classifier(noise = False, show = True)

    # Introducimos ruido en las etiquetas
    y_noise = simulate_noise(y, 0.1)

    # Mostramos el resultado de la nueva clasificación
    scatter_plot(X, ["x", "y"], y_noise,
        [[-b, -a, 1, 0, 0, 0]], ["Recta y = {:0.5f}x + ({:0.5f})".format(a, b)],
        title = "Clasificación dada por una recta (con ruido)")

#
# EJERCICIO 3: FRONTERA DE CLASIFICACIÓN PARA CLASIFICADORES CUADRÁTICOS
#

def ex3():
    """Dibuja la frontera de clasificación de una serie de
       clasificadores cuadráticos, junto con una nube de puntos
       clasificados según la recta del ejercicio anterior."""

    # Repetimos la segunda parte del ejercicio anterior
    X, y, a, b = random_line_classifier(noise = True)

    # Listamos los clasificadores para comparar
    classifiers = [
        [-b, -a, 1, 0, 0, 0],
        [100, -20, -40, 0, 1, 1],
        [50, 10, -40, 0, 0.5, 1],
        [-750, -10, -40, 0, 0.5, -1],
        [3, -5, 1, 0, -20, 0]]

    classifiers_names = [
        "f(x,y = y - {:0.5f}x - {:0.5f}".format(a, b),
        r"$f(x, y) = (x-10)^2 + (y-20)^2 - 400$",
        r"$f(x, y) = 0.5(x+10)^2 + (y-20)^2 - 400$",
        r"$f(x, y) = 0.5(x-10)^2 - (y+20)^2 - 400$",
        r"$f(x, y) = y - 20x^2 -5x +3$"]

    # Mostramos las gráficas
    for i, (cl, cl_name) in enumerate(zip(classifiers, classifiers_names)):
        scatter_plot(X, ["x", "y"], y, [cl], [cl_name],
            title = "Frontera de clasificación para el clasificador " + str(i + 1))

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
    #ex2()
    print("\n--- EJERCICIO 3 ---")
    ex3()

if __name__ == "__main__":
    main()
