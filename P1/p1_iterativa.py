#!/usr/bin/env python
# coding: utf-8
# uso: ./p1_iterativa.py

##########################################################################
# Aprendizaje Automático. Curso 2019/20.
# Práctica 1: Programación.
# Ejercicio sobre búsqueda iterativa de óptimos.
# Antonio Coín Castro. Grupo 3.
##########################################################################

#
# LIBRERÍAS
#

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior."""

    input("(Pulsa [Enter] para continuar...)\n")
    plt.close()

#
# EJERCICIO 1: IMPLEMENTACIÓN DE GRADIENTE DESCENDENTE
#

def gd(f, df, w, lr, max_it, eps = -np.inf):
    """Implementación del algoritmo iterativo de gradiente descendente.
         - f = f(x, y): función real-valuada a optimizar.
         - df = df(x, y): gradiente de f.
         - w = (u, v): punto inicial.
         - lr: valor del learning rate.
         - max_it: número máximo de iteraciones (criterio de parada).
         - eps: mínimo valor admisible de f (criterio de parada opcional).

       Devuelve:
         - valor del mínimo encontrado.
         - número de iteraciones realizadas.
         - lista con la evolución de los puntos en el algoritmo."""

    w_ = w  # No modificamos el parámetro w
    it = 0
    evol = [w_]

    while it < max_it and f(*w_) > eps:
        w_ = w_ - lr * df(*w_)
        evol += [w_]
        it += 1

    return w_, it, evol

#
# EJERCICIO 2: MINIMIZACIÓN DE LA FUNCIÓN E(u, v)
#

def E(u, v):
    """Función E(u, v) del ejercicio 2."""

    return (u * np.exp(v) - 2 * v * np.exp(-u)) ** 2

def dEu(u, v):
    """Derivada parcial de la función E(u,v) con respecto a u."""

    return (2 * (u * np.exp(v) - 2 * v * np.exp(-u))
        * (np.exp(v) + 2 * v * np.exp(-u)))

def dEv(u, v):
    """Derivada parcial de la función E(u,v) con respecto a v."""

    return (2 * (u * np.exp(v) - 2 * v * np.exp(-u))
        * (u * np.exp(v) - 2 * np.exp(-u)))

def dE(u, v):
    """Gradiente de la función E(u, v)."""

    return np.array([dEu(u, v), dEv(u, v)])

def ex2():
    """Ejecución de los distintos apartados del ejercicio 2."""

    # Fijamos los parámetros de ejecución
    w = np.array([1.0, 1.0], dtype = np.float64)
    lr = 0.1
    eps = 1e-14
    max_it = np.inf

    # Ejecutamos el algoritmo de gradiente descendente
    wmin, it, _ = gd(E, dE, w, lr, max_it, eps)

    # Mostramos los resultados que contestan a los apartados b) y c)
    print(f"  Punto inicial: ({w[0]}, {w[1]})")
    print("  Tasa de aprendizaje:", lr)
    print("  Tolerancia:", eps)
    print("  Número de iteraciones:", it)
    print(f"  Mínimo: ({wmin[0]}, {wmin[1]})")
    print("  Valor del mínimo:", E(*wmin), "\n")

#
# EJERCICIO 3: MINIMIZACIÓN DE LA FUNCIÓN f(x, y)
#

def surface_plot(f, fname, xmin, xmax, ymin, ymax, ws = []):
    """Muestra una figura 3D de la superficie dada por z = f(x, y),
       eventualmente junto con una serie de puntos destacados.
         - f: función a considerar.
         - fname: nombre de la función.
         - xmin, xmax: límites del plot en el eje X.
         - ymin, ymax: límites del plot en el eje Y.
         - ws: lista de puntos (2D) destacados."""

    # Establecemos el rango y las variables del plot
    x = np.linspace(xmin, xmax, 50)
    y = np.linspace(ymin, ymax, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Dibujamos la superficie
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(fname + "(x, y)")
    ax.plot_surface(X, Y, Z, alpha = 0.5)

    # Dibujamos los puntos destacados
    for w in ws:
        w = w[:, np.newaxis]
        ax.plot(*w, f(*w), 'ro', markersize = 10)

    plt.show(block = True)
    wait()

def contour_plot(f, xmin, xmax, ymin, ymax, w = None):
    """Pinta el diagrama de contorno para la función 'f = f(x, y)',
       posiblemente junto a un punto destacado 'w', en el
       entorno [xmin, xmax] x [ymin, ymax]."""

    # Establecemos el rango del plot
    x = np.arange(-2, 2, 0.01)
    y = np.arange(-2, 2, 0.01)

    # Pintamos el diagrama y el punto del mínimo
    fig = plt.figure()
    xx, yy = np.meshgrid(x, y, sparse = True)
    z = f(xx, yy)
    cont = plt.contourf(x, y, z)
    fig.colorbar(cont)
    if w is not None:
        plt.plot(*w, 'r*', markersize = 5)

    # Información del plot
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show(block = False)
    wait()

def f(x, y):
    """Función f(x, y) del ejercicio 3."""

    return ((x - 2) ** 2 + 2 * (y + 2) ** 2
        + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))

def dfx(x, y):
    """Derivada parcial de la función f(x, y) con respecto a x."""

    return (2 * (x - 2)
        + 4 * np.pi * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * x))

def dfy(x, y):
    """Derivada parcial de la función f(x, y) con respecto a y."""

    return (4 * (y + 2)
        + 4 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))

def df(x, y):
    """Gradiente de la función f(x, y)."""

    return np.array([dfx(x, y), dfy(x, y)])

def ap3A():
    """Ejecución del apartado a) del ejercicio 3."""

    # Fijamos los parámetros de las ejecuciones
    w = np.array([1.0, -1.0])
    max_it = 50

    # Nombres de los ejes para la gráfica
    plt.xlabel("Iteraciones")
    plt.ylabel("Valor de f(w)")

    for lr in [0.01, 0.1]:
        # Ejecutamos el algoritmo de gradiente descendente
        wmin, _, evol = gd(f, df, w, lr, max_it)

        # Imprimimos los resultados
        print(f"    Punto inicial: ({w[0]}, {w[1]})")
        print("    Tasa de aprendizaje:", lr)
        print("    Número de iteraciones:", max_it)
        print(f"    Mínimo: ({wmin[0]}, {wmin[1]})")
        print("    Valor del mínimo:", f(*wmin), "\n")

        # Mostramos la gráfica con la evolución del valor de la función
        plt.plot(range(max_it + 1), [f(*w) for w in evol],
            'o', label = r"$\eta$ = " + str(lr), linestyle = '--')

    plt.legend()
    plt.show(block = False)
    wait()

def ap3B():
    """Ejecución del apartado b) del ejercicio 3."""

    # Fijamos los parámetros de las ejecuciones
    max_it = 50
    lr = 0.01
    w_lst = [np.array([2.1, -2.1]),
        np.array([3.0, -3.0]),
        np.array([1.5, 1.5]),
        np.array([1.0, -1.0])]

    print("    Tasa de aprendizaje:", lr)
    print("    Número de iteraciones:", max_it)
    print("\n    {:^12}  {:^25}  {:^17}".format("Inicial", "Mínimo", "Valor"))

    for w in w_lst:
        # Ejecutamos el algoritmo de gradiente descendente
        wmin, _, _ = gd(f, df, w, lr, max_it)

        # Imprimimos los resultados
        print("    {}    {}      {: 1.5f}".format(w, wmin, f(*wmin)))

def ex3():
    """Ejecución de los distintos apartados del ejercicio 3."""

    print("  --- Apartado a)")
    ap3A()
    print("  --- Apartado b)")
    ap3B()

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Función principal. Ejecuta el ejercicio paso a paso."""

    # Número de decimales fijo para salida de vectores
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

    print("-------- EJERCICIO SOBRE BÚSQUEDA ITERATIVA DE ÓPTIMOS --------")
    print("--- EJERCICIO 2 ---")
    ex2()
    wait()
    print("\n--- EJERCICIO 3 ---")
    ex3()

if __name__ == "__main__":
    main()
