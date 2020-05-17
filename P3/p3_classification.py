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
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LassoCV, LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
from sklearn.ensemble import RandomForestClassifier
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
    """Ajuste de un modelo lineal para resolver un problema de clasificación."""

    # Cargamos los datos de entrenamiento y test
    print("Cargando datos de entrenamiento y test... ", end = "")
    X_train_pre, y_train = read_data(PATH + "optdigits.tra")
    X_test_pre, y_test = read_data(PATH + "optdigits.tes")
    print("Hecho.\n")

    """
    Pipeline 1: var2 + standardize2 + SelectFromModel(LassoCV(cv=5), th = 0.05)
        + poly2 + standardize + RL (Cs = 3, cv=5, L2)
        Test_acc: 98.386%
    Pipeline 2: PCA(0.95) + poly2 + standardize + RL (Cs=3, cv=5, L2)
        Test_ac: 98.720%
    """

    # Eliminamos variables con varianza < 0.1
    var2 = ("Eliminar varianza 0", VarianceThreshold(0.1))

    # Estandarizamos en origen y escala
    standardize2 = ("Estandarización 2", StandardScaler())

    # Hacemos selección de variables + whitening (?)
    selection = ("PCA", PCA(n_components = 0.95))
    selection2 = ("Regresión L1 para selección",
        SelectFromModel(
            LassoCV(cv = StratifiedKFold(5), n_jobs = -1),
            threshold = 0.05))

    # Transformamos a características polinómicas de grado 2
    poly2 = ("Polinomios grado 2", PolynomialFeatures(2))

    # Estandarizamos en origen y escala
    standardize = ("Estandarización", StandardScaler())

    # Juntamos todo el preprocesado en un pipeline
    preproc = Pipeline([selection, poly2, standardize])

    # Preprocesamos los datos de entrenamiento y test
    X_train = preproc.fit_transform(X_train_pre, y_train)
    X_test = preproc.transform(X_test_pre)

    # Elegimos los modelos y sus parámetros para CV
    models = [
        ("Logistic Regression", LogisticRegression(penalty = 'l2', max_iter = 500)),
        ("SGD + Hinge", SGDClassifier(loss = 'hinge')),
        ("Ridge", RidgeClassifier()),
        ("PLA", Perceptron())]
    params_lst = [
        {"C": [1e-4, 1.0, 1e4]},
        {"alpha": [1e-6, 1e-5, 1e-3, 1e-1, 1e2]},
        {"alpha": [0.01, 0.1, 1.0, 10.0]},
        {"penalty": ['l1', 'l2'], "alpha": [1e-6, 1e-4, 1e-2, 1.0]},
    ]

    for (name, model), params in zip(models, params_lst):
        # Buscamos los mejores parámetros por CV
        clf = GridSearchCV(model, params, scoring = 'accuracy', n_jobs = -1, cv = 5)

        # Entrenamos el mejor modelo
        start = default_timer()
        clf.fit(X_train, y_train)
        elapsed = default_timer() - start

        # Mostramos los resultados
        print("--- {} ---".format(name))
        print("Mejores parámetros: {}".format(clf.best_params_))
        print("Accuracy en training: {:.3f}%".format(
            100.0 * clf.score(X_train, y_train)))
        print("Accuracy en test: {:.3f}%".format(
            100.0 * clf.score(X_test, y_test)))
        print("Tiempo: {:.3f}s\n".format(elapsed))

    # Comparación con modelo no lineal
    start = default_timer()
    clf = RandomForestClassifier(n_estimators = 200, n_jobs = -1).fit(X_train, y_train)
    elapsed = default_timer() - start

    print("--- Random Forest (n = 200) ---")
    print("Accuracy en training: {:.3f}%".format(
        100.0 * clf.score(X_train, y_train)))
    print("Accuracy en test: {:.3f}%".format(
        100.0 * clf.score(X_test, y_test)))
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
