#!/usr/bin/env python
# coding: utf-8
# uso: ./p1_bonus.py

##########################################################################
# Aprendizaje Automático. Curso 2019/20.
# Práctica 1: Programación.
# Ejercicio sobre el método de Newton (bonus)
# Antonio Coín Castro. Grupo 3.
##########################################################################

#
# LIBRERÍAS
#

import numpy as np
from matplotlib import pyplot as plt

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior."""

    input("\n(Pulsa [Enter] para continuar...)\n")
    plt.close()

#
# IMPLEMENTACIÓN DEL MÉTODO DE NEWTON
#

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

def hfxx(x, y):
    """Derivada con respecto a x dos veces de f(x, y)."""

    return (2
        - 8 * np.pi ** 2 * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * x))

def hfyy(x, y):
    """Derivada con respecto a y dos veces de f(x, y)."""

    return (4
        - 8 * np.pi ** 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))

def hfxy(x, y):
    """Derivada cruzada de la función f(x,y)."""

    return 8 * np.pi ** 2 * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * x)

def hf(x, y):
    """Matriz hessiana de la función f(x, y)."""

    return np.array([[hfxx(x, y), hfxy(x, y)],
        [hfxy(x, y), hfyy(x, y)]])

def newton(f, df, hf, w, lr, max_it):
    """Implementación del método de Newton para optimización.
         - f = f(x, y): función real-valuada a optimizar.
         - df = df(x, y): gradiente de f.
         - hf = hf(x, y): matriz hessiana de f.
         - w = (u, v): punto inicial.
         - lr: valor del learning rate.
         - max_it: número máximo de iteraciones (criterio de parada).

       Devuelve:
         - valor del mínimo encontrado.
         - lista con la evolución de los puntos en el algoritmo."""

    w_ = w  # No modificamos el parámetro w
    it = 0
    evol = [w_]

    while it < max_it:
        w_ = w_ - lr * np.linalg.inv(hf(*w_)) @ df(*w_)
        evol += [w_]
        it += 1

    return w_, evol

def bonus():
    """Implementación del bonus sobre el método de Newton."""

    # Fijamos los parámetros
    max_it = 50
    w_lst = [np.array([2.1, -2.1]),
         np.array([3.0, -3.0]),
         np.array([1.5, 1.5]),
         np.array([1.0, -1.0])]

    for lr in [0.1, 1]:
        plt.xlabel("Iteraciones")
        plt.ylabel("Valor de f(w)")
        print("  Número de iteraciones:", max_it)
        print("  Tasa de aprendizaje:", lr)

        for w in w_lst:
            # Ejecutamos el algoritmo
            wmin, evol = newton(f, df, hf, w, lr, max_it)

            # Mostramos los resultados
            print(f"\n  Punto inicial: ({w[0]}, {w[1]})")
            print(f"  Mínimo: ({wmin[0]}, {wmin[1]})")
            print("  Valor del mínimo: {0:0.5f}".format(f(*wmin)))

            # Mostramos la gráfica con la evolución del valor de la función
            plt.plot(range(max_it + 1), [f(*w) for w in evol],
                'o', label = "w = " + str(w), linestyle = '--')

        plt.title("Learning rate " + r"$\eta = $" + str(lr))
        plt.legend()
        plt.show(block = False)
        wait()

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Función principal. Ejecuta el ejercicio paso a paso."""

    print("-------- EJERCICIO DE BONUS --------")
    print("--- MÉTOODO DE NEWTON ---")
    bonus()

if __name__ == "__main__":
    main()
