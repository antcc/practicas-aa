#!/usr/bin/env python
# coding: utf-8
# uso: ./p1_regresion.py

##########################################################################
# Aprendizaje Automático. Curso 2019/20.
# Práctica 1: Programación.
# Ejercicio sobre regresión lineal.
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
CLASSES = [1, 5]
LABEL_C1 = -1
LABEL_C2 = 1

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior."""

    input("\n(Pulsa [Enter] para continuar...)\n")
    plt.close()

def read_data(file_X, file_y):
    """Leer archivos de características y clases en un par de
       vectores. El formato de salida de las características es
       [1, x_1, x_2].
         - file_X: archivo con las características bidimensionales.
         - file_y: archivo con las clases."""

    # Leemos los ficheros
    data_X = np.load(file_X)
    data_y = np.load(file_y)
    X = []
    y = []

    # Solo guardamos los datos cuya clase sea la 1 o la 5
    for i in range(0, data_y.size):
        if data_y[i] in CLASSES:
            if data_y[i] == CLASSES[0]:
                y.append(LABEL_C1)
            else:
                y.append(LABEL_C2)
            X.append(np.array([1, data_X[i][0], data_X[i][1]]))

    X = np.array(X, np.float64)
    y = np.array(y, np.float64)

    return X, y

def scatter_plot(X, axis, y = None, ws = None, ws_labels = None, is_linear = True):
    """Muestra un scatter plot con leyenda y (opcionalmente) rectas
       de regresión.
         - X: matriz de características con primera componente 1, y
           con exactamente dos grados de libertad.
         - axis: nombres de los ejes.
         - y: clases.
         - ws: vectores de pesos
         - ws_labels: etiquetas de los vectores de pesos. Debe
           aparecer obligatoriamente si 'ws' no es vacío.
         - is_linear: controla si las características son lineales."""

    # Establecemos límites y tamaño del plot
    plt.figure(figsize = (8, 6))
    xmin, xmax = np.min(X[:, 1]), np.max(X[:, 1])
    ymin, ymax = np.min(X[:, 2]), np.max(X[:, 2])
    plt.xlim(xmin - 0.1, xmax + 0.1)
    plt.ylim(ymin - 0.1, ymax + 0.1)

    # Asignamos vector de etiquetas
    if y is None:
        c = [1 for _ in range(len(X))]
    else:
        c = y

    # Mostramos scatter plot con leyenda
    scatter = plt.scatter(X[:, 1], X[:, 2], c = c,
        cmap = ListedColormap(['r', 'lime']), edgecolors = 'k')
    if y is not None:
        legend1 = plt.legend(
            *scatter.legend_elements(),
            title = "Clases",
            loc = "upper right")

    plt.xlabel(axis[0])
    plt.ylabel(axis[1])

    # Mostramos modelos ajustados
    if ws is not None:
        if is_linear:  # Pintamos una recta
            x = np.array([xmin, xmax])
            for w, l in zip(ws, ws_labels):
                plt.plot(x, (-w[0] - w[1] * x) / w[2], label = l, linewidth = 2)
        else:  # Pintamos una cónica
            xx, yy = np.meshgrid(np.linspace(xmin - 0.1, xmax + 0.2, 100),
                np.linspace(ymin - 0.1, ymax + 0.1, 100))

            # Función que encapsula el producto escalar <w^T, X>
            h = lambda x, y, w: np.array([1, x, y, x * y, x * x, y * y, x**3, y**3, (x**2)*y, (y**2)*x]).dot(w)

            # Pintamos la curva de nivel 0 en el plano
            for w, l in zip(ws, ws_labels):
                z = h(xx, yy, w)
                plt.contour(xx, yy, z, levels = [0]).collections[0].set_label(l)

        plt.legend(loc = "lower right")

    # Añadimos leyenda
    if y is not None:
        plt.gca().add_artist(legend1)

    plt.show(block = False)
    wait()

#
# EJERCICIO 1: AJUSTE DE MODELOS DE REGRESIÓN LINEAL
#

def sgd(X, y, lr, batch_size, max_it):
    """Implementación del algoritmo de gradiente descendente estocástico. Devuelve
       el vector de pesos encontrado.
         - X: matriz con vectores de características con primera componente 1.
         - y: vector de etiquetas.
         - lr: valor del learning rate.
         - max_it: número máximo de iteraciones (criterio de parada).
         - batch_size: tamaño del minibatch."""

    it = 0
    start = 0  # Contador de batch
    n = len(X)
    dims = len(X[0])
    idxs = np.arange(n)  # Vector de índices
    w = np.zeros((dims,))  # Punto inicial w = 0

    while it < max_it:
        # Barajamos los datos en cada pasada completa
        if start == 0:
            np.random.shuffle(idxs)

        # Procesamos cada minibatch de forma secuencial
        idx = idxs[start:start + batch_size]

        # Actualizamos el vector de pesos
        w = (w - lr * (2 / batch_size)
            * (X[idx].T.dot(X[idx].dot(w) - y[idx])))

        # Actualizamos contadores
        it += 1
        start = start + batch_size
        if start > n:
            start = 0

    return w

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

def err(w, X, y):
    """Expresión del error cometido por un modelo de regresión lineal.
         - w: vector de pesos.
         - X: matriz de características con primera componente 1.
         - y: vector de etiquetas."""

    return 1 / len(X) * ((X.dot(w) - y) ** 2).sum()

def ex1():
    """Ajuste de dos modelos de regresión lineal, usando SGD y el
       método de la pseudoinversa."""

    # Cargamos los datos
    X_train, y_train = read_data(PATH + "X_train.npy", PATH + "y_train.npy")
    X_test, y_test = read_data(PATH + "X_test.npy", PATH + "y_test.npy")

    # Estimamos un modelo SGD y otro con pseudoinversa
    w_sgd = sgd(X_train, y_train, 0.1, 64, 100)
    w_pseudo = pseudoinverse(X_train, y_train)

    # Mostramos los resultados
    print("  Vector de pesos con SGD:", w_sgd)
    print("  Vector de pesos con pseudoinversa:", w_pseudo)
    scatter_plot(X_train, ["Intensidad promedio", "Simetría"],
        y_train, [w_sgd, w_pseudo], ["SGD", "Pseudoinversa"])

    # Mostramos los errores
    print("  Errores con SGD:")
    print("    E_in:", err(w_sgd, X_train, y_train))
    print("    E_out:", err(w_sgd, X_test, y_test))

    print("\n  Errores con Pseudoinversa:")
    print("    E_in:", err(w_pseudo, X_train, y_train))
    print("    E_out:", err(w_pseudo, X_test, y_test))

#
# EJERCICIO 2: EVOLUCIÓN DEL ERROR CON LA COMPLEJIDAD DEL MODELO
#

def uniform_data(n, d, size):
    """Genera 'n' puntos de dimension 'd' uniformente distribuidos en el hipercubo
       definido por [-size, size]."""

    return np.random.uniform(-size, size, (n, d))

def f(x, y):
    """Función signo para asignar etiquetas."""

    return np.sign((x - 0.2) ** 2 + y * y - 0.6)

def generate_features(n, is_linear):
    """Genera características para el experimento del ejercicio 2.
         - n: número de puntos a generar.
         - is_linear: si es True, las características son [1, x_1, x_2]. En
           otro caso, son [1, x_1, x_2, x_1x_2, x_1^2, x_2^2]."""

    # Generamos una muestra de 'n' puntos en [-1, 1] x [-1, 1]
    X = uniform_data(n, 2, 1)

    # Formamos el vector de características
    X = np.hstack((np.ones((n, 1)), X))
    if not is_linear:
        X = np.vstack(
            (X.T,
            X[:, 1] * X[:, 2],
            X[:, 1] ** 2,
            X[:, 2] ** 2),
            X[:, 1] ** 3,
            X[:, 2] ** 3,
            (X[:, 1] ** 2) * X[:, 2],
            (X[:, 2] ** 2) * X[:, 1]).T

    # Generamos etiquetas y perturbamos aleatoriamente un 10%
    y = np.array([f(x[1], x[2]) for x in X])
    idxs = np.random.choice(n, int(0.1 * n), replace = False)
    y[idxs] = -y[idxs]

    return X, y

def experiment(is_linear, lr, show = False):
    """Realización del experimento descrito en los apartados a), b) y c) del
       ejercicio 2. Devuelve los errores E_in y E_out.
         - is_linear: controla si las características son lineales o no lineales.
         - lr: learning rate para SGD.
         - show: controla si se muestran por pantalla los resultados."""

    # Generamos datos de entrenamiento
    X, y = generate_features(1000, is_linear)

    if show:
        print("  Generados 1000 puntos de entrenamiento.")
        scatter_plot(X, ["x1", "x2"])
        print("  Generadas etiquetas con ruido.")
        scatter_plot(X, ["x1", "x2"], y)

    # Ajustamos un modelo de regresión lineal mediante SGD
    w = sgd(X, y, lr, 64, 100)

    # Generamos datos de test
    X_test, y_test = generate_features(1000, is_linear)

    # Calculamos los errores
    ein = err(w, X, y)
    eout = err(w, X_test, y_test)

    # Mostramos los resultados
    if show:
        model = ("SGD con características "
                 + ("no lineales" if not is_linear else "lineales."))
        print("  Vector de pesos:", w)
        print("  Errores en el experimento:")
        print("    E_in:", ein)
        print("    E_out:", eout)
        scatter_plot(X, ["x1", "x2"],
            y, [w], [model], is_linear)

    return [ein, eout]

def ex2():
    """Estudio de la bondad del ajuste lineal frente al aumento
       de complejidad del modelo."""

    # Número de ejecuciones del experimento
    N = 1000

    print("  Realizando experimento lineal "
          + "con características [1, x_1, x_2].")

    # Ejecutamos el experimento lineal una vez mostrando gráficas y resultados
    experiment(is_linear = True, lr = 0.1, show = True)

    # Realizamos el experimento lineal 1000 veces
    errors_l = np.array([0.0, 0.0])
    for _ in range(N):
        errors_l += experiment(is_linear = True, lr = 0.1)
    errors_l /= N

    print("  Errores medios en 1000 experimentos con características lineales:")
    print("    E_in:", errors_l[0])
    print("    E_out:", errors_l[1])

    print("\n  Realizando experimento no lineal "
          + "con características [1, x_1, x_2, x_1 * x_2, x_1^2, x_2^2].")

    # Ejecutamos el experimento no lineal una vez mostrando gráficas y resultados
    experiment(is_linear = False, lr = 0.3, show = True)

    # Realizamos el experimento no lineal 1000 veces
    errors_nl = np.array([0.0, 0.0])
    for _ in range(N):
        errors_nl += experiment(is_linear = False, lr = 0.3)
    errors_nl /= N

    print("  Errores medios en 1000 experimentos con características no lineales:")
    print("    E_in:", errors_nl[0])
    print("    E_out:", errors_nl[1])

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Función principal. Ejecuta el ejercicio paso a paso."""

    # Semilla aleatoria para reproducibilidad
    np.random.seed(SEED)

    # Número de decimales fijo para salida de vectores
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

    print("-------- EJERCICIO SOBRE REGRESIÓN LINEAL --------")
    print("--- EJERCICIO 1 ---")
    ex1()
    wait()
    print("\n--- EJERCICIO 2 ---")
    ex2()

if __name__ == "__main__":
    main()
