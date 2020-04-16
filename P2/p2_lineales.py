#!/usr/bin/env python
# coding: utf-8
# uso: ./p2_lineales.py

##########################################################################
# Aprendizaje Automático. Curso 2019/20.
# Práctica 2: Modelos lineales.
# Ejercicio sobre modelos lineales: PLA y regresión logística.
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
         - X: vector de características de la forma [1, x1, x2].
         - axis: nombres de los ejes.
         - y: vector de etiquetas o clases.
         - ws: lista de vectores 3-dimensionales que representan cada recta.
         - labels: lista de etiquetas de las rectas.
         - title: título del plot."""

    # Establecemos tamaño, colores e información del plot
    plt.figure(figsize = (8, 6))
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    cmap = ListedColormap(['r', 'lime'])
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

def uniform_sample(n, d, low, high):
    """Genera 'n' puntos de dimension 'd' uniformente distribuidos en el hipercubo
       definido por [low, high]."""

    return np.random.uniform(low, high, (n, d))

def line_sample(low, high):
    """Simula de manera uniforme los parámetros de una recta que
       corte al cuadrado [low, high] x [low, high]. Se devuelve un vector w
       que representa la recta w0 + w1x + y = 0."""

    # Generamos uniformemente dos puntos en el cuadrado
    points = np.random.uniform(low, high, size = (2, 2))
    x1, x2 = points[:, 0]
    y1, y2 = points[:, 1]

    # Calculamos la pendiente y el término independiente
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    return np.array([-b, -a, 1])

def sign(x):
    """Devuelve el signo de x, considerando que el signo de 0 es 1."""

    return 1 if x >= 0 else -1

def f(x, y, w):
    """Función utilizada para asignar etiquetas binarias a los puntos.
       Mide el signo de la distancia del punto (x, y) a la recta dada por
       el vector 'w'. Se considera que los puntos que están sobre la recta
       pertenecen a la clase del 1."""

    return sign(w[0] + w[1] * x + w[2] * y)

def simulate_noise(y, p):
    """Introduce ruido en el vector de etiquetas binarias 'y', cambiando una
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

def random_line_classifier(n):
    """Genera una nube de puntos de forma uniforme y una recta
       a partir de la cual clasificarlos, con eventual ruido.
       Devuelve los puntos, la recta y las etiquetas generadas (con y sin ruido).
         - n: número de puntos a generar."""

    # Generamos 100 puntos y una recta en [-50, 50] x [-50, 50]
    X = uniform_sample(n, 2, -50, 50)
    v = line_sample(-50, 50)

    # Etiquetamos los puntos según el signo de la distancia a la recta
    y = np.array([f(x[0], x[1], v) for x in X])

    # Introducimos ruido en las etiquetas
    y_noise = simulate_noise(y, 0.1)

    return X, y, y_noise, v

def predict(X, y, w):
    """Devuelve las métricas de accuracy y balanced accuracy (en %) sobre
       los puntos homogéneos de 'X' para una recta dada por el vector 'w',
       siendo el vector 'y' el de las verdaderas etiquetas."""

    fun = lambda x: x.dot(w)
    y_pred = [sign(fun(x)) for x in X]
    acc = 100 * accuracy_score(y, y_pred, normalize = True)
    balanced_acc = 100 * balanced_accuracy_score(y, y_pred)

    return acc, balanced_acc

#
# EJERCICIO 1: ALGORITMO PERCEPTRÓN
#

def fit_pla(X, y, max_it, w_ini):
    """Ajusta los parámetros de un hiperplano que separe datos en dos clases
       usando el algoritmo perceptrón. Devuelve también el número de iteraciones.
         - X: matriz de datos en coordenadas homogéneas (primera componente 1).
         - y: vector de etiquetas (1 ó -1).
         - max_it: número máximo de iteraciones.
         - w_ini: punto inicial del algoritmo (array de numpy de tipo float)."""

    w = w_ini.copy()  # No modificamos el parámetro w_ini

    # Repetimos el algoritmo el número de iteraciones deseado
    for it in range(max_it + 1):
        change = False

        # Recorremos todos los ejemplos
        for x, l in zip(X, y):
            # Si está mal clasificado, actualizamos los pesos
            if sign(x.dot(w)) != l:
                w += l * x
                change = True

        # Si no ha habido cambios, hemos terminado
        if not change:
            break

    return w, it

def ex1():
    """Ejemplos de ajuste con el algoritmo perceptrón sobre datos
       linealmente separables y datos no linealmente separables (con ruido)."""

    # Establecemos de nuevo la semilla para recuperar los puntos
    # y la recta del ejercicio anterior
    np.random.seed(SEED)

    # Obtenemos los datos, la recta que los separa y las etiquetas con y sin ruido
    N = 100
    X, y_orig, y_noise, v = random_line_classifier(N)

    # Convertimos a coordenadas homogéneas
    X = np.hstack((np.ones((N, 1)), X))

    # Ejecutamos el algoritmo PLA sobre los datos con y sin ruido
    for y, ap in zip([y_orig, y_noise], ["a", "b"]):
        print("--- Apartado {}): etiquetas {} ruido"
            .format(ap, "sin" if ap == "a" else "con"))

        # Resultados de la recta original
        acc, balanced_acc = predict(X, y, v)
        print("Recta original")
        print("    Accuracy = {:0.3f}%".format(acc))
        print("    Balanced accuracy = {:0.3f}%\n".format(balanced_acc))

        # Tomando como punto inicial el 0
        max_it = 1000
        v_ini = np.zeros(3)
        v_pla, it = fit_pla(X, y, max_it, v_ini)

        # Mostramos los resultados
        acc_pla, balanced_acc_pla = predict(X, y, v_pla)
        print("Recta encontrada por PLA con vector inicial 0")
        print("    Iteraciones:", it)
        print("    Accuracy = {:0.3f}%".format(acc_pla))
        print("    Balanced accuracy = {:0.3f}%\n".format(balanced_acc_pla))

        # Tomando como punto inicial uno aleatorio en [0, 1]
        K = 10
        it_acum = acc_pla_acum = balanced_acc_pla_acum = 0
        for _ in range(K):
            v_pla, it = fit_pla(X, y, max_it, np.random.rand(3))
            acc_pla, balanced_acc_pla = predict(X, y, v_pla)
            it_acum += it
            acc_pla_acum += acc_pla
            balanced_acc_pla_acum += balanced_acc_pla

        # Mostramos los resultados
        print("Recta encontrada por PLA con vector inicial aleatorio (de media)")
        print("    Iteraciones:", it_acum / K)
        print("    Accuracy = {:0.3f}%".format(acc_pla_acum / K))
        print("    Balanced accuracy = {:0.3f}%".format(balanced_acc_pla_acum / K))

        # Mostramos la recta original y una encontrada por PLA
        scatter_plot(X, ["x", "y"], y, [v, v_pla],
            ["Recta y = {:0.3f}x + ({:0.3f}) (original)".format(*(-v[:2] / v[2])[::-1]),
            "Recta y = {:0.3f}x + ({:0.3f}) (PLA)".format(*(-v_pla[:2] / v_pla[2])[::-1])],
            title = "Recta de separación original y dada por PLA ({} ruido)".format(
                "sin" if ap == "a" else "con"))

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Función principal. Ejecuta el ejercicio paso a paso."""

    # Semilla aleatoria para reproducibilidad
    np.random.seed(SEED)

    # Número de decimales fijo para salida de vectores
    np.set_printoptions(formatter = {'float': lambda x: "{:0.3f}".format(x)})

    print("-------- EJERCICIO SOBRE MODELOS LINEALES --------\n")
    print("--- EJERCICIO 1: ALGORITMO PLA ---")
    ex1()
    print("--- EJERCICIO 2: REGRESIÓN LOGÍSTICA ---")
    #ex2()

if __name__ == "__main__":
    main()
