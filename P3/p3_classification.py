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
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
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
    #np.random.shuffle(data)
    return data[:, :-1], data[:, -1]

#
# PROBLEMA DE CLASIFICACIÓN EN OPTDIGITS
#

def preprocess_pipeline(selection_strategy = 0):
    """Construye una lista de transformaciones y parámetros para el
       preprocesamiento de datos. La estrategia de selección de variables
       se controla mediante el parámetro 'selecion_strategy':
         * 0: Mediante PCA
         * 1: Mediante regresión L1 eliminando los coeficientes que vayan a 0.
         * 2: Mediante información conjunta.
       Cualquier otro valor hará que no se haga selección de variables."""

    # Selección con PCA
    if selection_strategy == 0:
        preproc = [
            ("selection", PCA(0.95)),
            ("poly", PolynomialFeatures(2)),
            ("standardize2", StandardScaler())]

        preproc_params = {}

    # Selección con Lasso
    elif selection_strategy == 1:
        preproc = [
            ("var", VarianceThreshold(0.01)),
            ("standardize", StandardScaler()),
            ("selection", SelectFromModel(Lasso())),
            ("poly", PolynomialFeatures(2)),
            ("standardize2", StandardScaler())]

        preproc_params = {
            "selection__threshold": [0.005, 0.05],
            "selection__estimator__alpha": np.logspace(-5, 1, 3)}

    # Selección con información mutua
    elif selection_strategy == 2:
        preproc = [
            ("var", VarianceThreshold(0.01)),
            ("standardize", StandardScaler()),
            ("selection", SelectKBest()),
            ("poly", PolynomialFeatures(2)),
            ("standardize2", StandardScaler())]

        preproc_params = {
            "selection__score_func": [mutual_info_classif, f_classif],
            "selection__k": [35, 45, 55]}

    # Sin preprocesamiento
    else:
        preproc = [
            ("var", VarianceThreshold(0.01)),
            ("poly", PolynomialFeatures(2)),
            ("standardize2", StandardScaler())]

        preproc_params = {}

    return preproc, preproc_params

def classification_fit():
    """Ajuste de un modelo lineal para resolver un problema de clasificación."""

    """
    Pipeline 1: var2 + standardize2 + SelectFromModel(LassoCV(cv=5), th = 0.05)
        + poly2 + standardize + RL (Cs = 3, cv=5, L2)
        Test_acc: 98.386%
    Pipeline 2: PCA(0.95) + poly2 + standardize + RL (Cs=3, cv=5, L2)
        Test_ac: 98.720%
    """

    # Cargamos los datos de entrenamiento y test
    print("Cargando datos de entrenamiento y test... ", end = "")
    start = default_timer()
    X_train, y_train = read_data(PATH + "optdigits.tra")
    X_test, y_test = read_data(PATH + "optdigits.tes")
    print("Hecho.\n")

    # Construimos un pipeline de preprocesado + clasificación
    preproc, preproc_params = preprocess_pipeline(selection_strategy = 0)
    pipe = Pipeline(preproc + [("clf", LogisticRegression())])

    # Elegimos los modelos lineales y sus parámetros para CV
    search_space = [
        {**preproc_params,
         **{"clf": [LogisticRegression(penalty = 'l2', max_iter = 500)],
            "clf__C": np.logspace(-4, 4, 3)}},
        {**preproc_params,
         **{"clf": [SGDClassifier(loss = 'hinge', random_state = SEED)],
            "clf__penalty": ['l1', 'l2'],
            "clf__alpha": np.logspace(-6, 0, 3)}},
        {**preproc_params,
         **{"clf": [RidgeClassifier(random_state = SEED)],
            "clf__alpha": np.logspace(-2, 2, 3)}},
        {**preproc_params,
         **{"clf": [Perceptron(random_state = SEED)],
            "clf__penalty": ['l1', 'l2'],
            "clf__alpha": np.logspace(-6, 0, 3)}}]

    # Buscamos los mejores parámetros por CV
    start = default_timer()
    best_clf = GridSearchCV(pipe, search_space, scoring = 'accuracy',
            cv = 5, n_jobs = -1)
    best_clf.fit(X_train, y_train)
    elapsed = default_timer() - start

    # Mostramos los resultados
    print("--- Mejor clasificador lineal ---")
    print("Parámetros:\n{}".format(best_clf.best_params_))
    print("Accuracy en CV: {:.3f}%".format(100.0 * best_clf.best_score_))
    print("Accuracy en training: {:.3f}%".format(
        100.0 * best_clf.score(X_train, y_train)))
    print("Accuracy en test: {:.3f}%".format(
        100.0 * best_clf.score(X_test, y_test)))
    print("Tiempo: {:.3f}s\n".format(elapsed))

    # Elegimos un modelo no lineal y sus parámetros para CV
    nonlinear_search_space = [
        {**preproc_params,
         **{"clf": [RandomForestClassifier(random_state = SEED)],
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [30, None]}}]

    # Buscamos los mejores parámetros por CV
    start = default_timer()
    nonlinear_best_clf = GridSearchCV(pipe, nonlinear_search_space,
        scoring = 'accuracy', cv = 5, n_jobs = -1)
    nonlinear_best_clf.fit(X_train, y_train)
    elapsed = default_timer() - start

    # Mostramos los resultados
    print("--- Mejor clasificador no lineal ---")
    print("Parámetros:\n{}".format(nonlinear_best_clf.best_params_))
    print("Accuracy en CV: {:.3f}%".format(100.0 * nonlinear_best_clf.best_score_))
    print("Accuracy en training: {:.3f}%".format(
        100.0 * nonlinear_best_clf.score(X_train, y_train)))
    print("Accuracy en test: {:.3f}%".format(
        100.0 * nonlinear_best_clf.score(X_test, y_test)))
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
