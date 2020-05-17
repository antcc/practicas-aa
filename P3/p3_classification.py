#!/usr/bin/env python
# coding: utf-8
# uso: ./p3_classification.py

"""
Aprendizaje Automático. Curso 2019/20.
Práctica 3: Ajuste de modelos lineales.
Intentamos conseguir el mejor ajuste con modelos lineales para un
problema de clasificación y otro de regresión.

Antonio Coín Castro. Grupo 3.
"""

#
# LIBRERÍAS
#

import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.pipeline import Pipeline

#
# PARÁMETROS GLOBALES
#

SEED = 2020
PATH = "datos/"
SAVE_FIGURES = False

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior."""

    input("\n(Pulsa [Enter] para continuar...)\n")
    plt.close()

def read_data(filename):
    """Lee los datos de dichero y los separa en características
       y etiquetas."""

    data = np.genfromtxt(filename, delimiter = ",", dtype = np.double)
    return data[:, :-1], data[:, -1]

#
# PROBLEMA DE CLASIFICACIÓN EN OPTDIGITS
#

def classification_fit():
    """Desarrolla la parte de ajustar un modelo lineal para resolver un
       problema de clasificación."""

    # Cargamos los datos de entrenamiento y test
    print("Cargando datos de entrenamiento y test... ", end = "")
    X_train, y_train = read_data(PATH + "optdigits.tra")
    X_test, y_test = read_data(PATH + "optdigits.tes")
    print("Hecho.\n")

    """
    Pipeline 1: var2 + standardize2 + SelectFromModel(LassoCV(cv=5), th = 1e-2)
        + poly2 + standardize + RL (Cs = 3, cv=5, L2)
        Test_acc: 98.275%
    Pipeline 2: PCA(0.95) + poly2 + standardize + RL (Cs=3, cv=5, L2)
        Test_ac: 98.720%
    """

    var2 = ("Eliminar varianza 0", VarianceThreshold(0.1))

    standardize2 = ("Estandarización 2", StandardScaler())

    # Hacemos selección de variables + whitening (?)
    selection = ("PCA", PCA(n_components = 0.95))
    selection2 = ("Regresión L1 para selección",
        SelectFromModel(
            LassoCV(cv = StratifiedKFold(5), n_jobs = -1),
            threshold = 1e-2))

    # Transformamos a características polinómicas de grado 2
    poly2 = ("Polinomios grado 2", PolynomialFeatures(2))

    # Normalizamos en origen y escala
    standardize = ("Estandarización", StandardScaler())

    # Elegimos un clasificador
    rl = ("Regresión logística",
        LogisticRegressionCV(Cs = 3, cv = 5, penalty = 'l2',
            scoring = 'accuracy', max_iter = 500, n_jobs = -1))

    # Juntamos preprocesado y clasificación
    classifier = Pipeline([var2, standardize2, selection2, poly2, standardize, rl])

    # Entrenamos el modelo
    start = default_timer()
    classifier.fit(X_train, y_train)
    elapsed = default_timer() - start

    # Mostramos los resultados
    print("Accuracy en training: {:.3f}%".format(
        100.0 * classifier.score(X_train, y_train)))
    print("Accuracy en test: {:.3f}%".format(
        100.0 * classifier.score(X_test, y_test)))
    print("Tiempo: {:.3f}s".format(elapsed))

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Función principal. Ejecuta el ejercicio paso a paso."""

    # Semilla aleatoria para reproducibilidad
    np.random.seed(SEED)

    # Número de decimales fijo para salida de vectores
    np.set_printoptions(formatter = {'float': lambda x: "{:0.3f}".format(x)})

    print("-------- AJUSTE DE MODELOS LINEALES --------")
    print("--- PARTE 1: CLASIFICACIÓN ---")
    classification_fit()

if __name__ == "__main__":
    main()
