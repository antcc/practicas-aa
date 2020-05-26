#!/usr/bin/env python
# coding: utf-8
# uso: ./p3_classification.py

"""
Aprendizaje Automático. Curso 2019/20.
Práctica 3: Ajuste de modelos lineales. Clasificación.
Intentamos conseguir el mejor ajuste con modelos lineales para un
problema de clasificación.

Base de datos: Optical Recognition of Handwritten Digits
https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits

Antonio Coín Castro. Grupo 3.
"""

#
# LIBRERÍAS
#

import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import (SelectFromModel, VarianceThreshold, SelectKBest,
    f_classif)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix

#
# PARÁMETROS GLOBALES
#

SEED = 2020
PATH = "datos/"
TMPDIR = "tmpdir"
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

def preprocess_pipeline(selection_strategy = 0):
    """Construye una lista de transformaciones y parámetros para el
       preprocesamiento de datos. La estrategia de selección de variables
       se controla mediante el parámetro 'selecion_strategy'."""

    # Reducción de dimensionalidad
    if selection_strategy == 0:
        preproc = [
            ("selection", PCA(0.95)),
            ("poly", PolynomialFeatures(2)),
            ("standardize", StandardScaler())]

    # Selección de variables
    else:
        preproc = [
            ("poly", PolynomialFeatures(2)),
            ("var", VarianceThreshold(0.1)),
            ("standardize", StandardScaler())]

    return preproc

def classification_fit(compare = False, selection_strategy = 0, show = True):
    """Ajuste de un modelo lineal para resolver un problema de clasificación.
       Opcionalmente se puede ajustar también un modelo no lineal (RandomForest)
       y un clasificador aleatorio para comparar el rendimiento.
         - compare_nonlinear: controla si se ajusta un modelo no lineal.
         - selection_strategy: estrategia de selección de características
             * 0: Mediante PCA.
             * 1: Mediante regresión L1 eliminando los coeficientes que vayan a 0.
             * 2: Mediante f-test.
         - show: controla si se muestran gráficas."""

    # Cargamos los datos de entrenamiento y test
    print("Cargando datos de entrenamiento y test... ", end = "")
    start = default_timer()
    X_train_pre, y_train = read_data(PATH + "optdigits.tra")
    X_test_pre, y_test = read_data(PATH + "optdigits.tes")
    print("Hecho.")

    # Preprocesamos los datos
    print("Preprocesando datos... ", end = "")
    preproc = preprocess_pipeline(selection_strategy)
    preproc_pipe = Pipeline(preproc)
    X_train = preproc_pipe.fit_transform(X_train_pre, y_train)
    X_test = preproc_pipe.transform(X_test_pre)
    print("Hecho.\n")

    # Construimos un pipeline de selección + clasificación
    pipe_lst = []
    if selection_strategy == 1:
        pipe_lst += [("var2", VarianceThreshold()),
                     ("selection", SelectFromModel(Lasso(max_iter = 1100, alpha = 0.01),
                        threshold = 0.001))]
    elif selection_strategy == 2:
        pipe_lst += [("var2", VarianceThreshold()),
                     ("selection", SelectKBest(f_classif, k = X_train.shape[1] // 3))]
    pipe = Pipeline(pipe_lst + [("clf", LogisticRegression())])

    # Elegimos los modelos lineales y sus parámetros para CV
    search_space = [
        {"clf": [LogisticRegression(penalty = 'l2', max_iter = 500)],
         "clf__C": np.logspace(-4, 4, 3)},
        {"clf": [SGDClassifier(loss = 'hinge', penalty = 'l2', random_state = SEED)],
         "clf__alpha": np.logspace(-6, 0, 3)},
        {"clf": [RidgeClassifier(random_state = SEED)],
         "clf__alpha": np.logspace(-6, 0, 3)}]

    # Buscamos los mejores parámetros por CV
    print("Realizando selección de modelos lineales... ", end = "")
    start = default_timer()
    best_clf = GridSearchCV(pipe, search_space, scoring = 'accuracy',
            cv = 5, n_jobs = -1)
    best_clf.fit(X_train, y_train)
    elapsed = default_timer() - start
    print("Hecho.\n")

    # Mostramos los resultados
    print("--- Mejor clasificador lineal ---")
    print("Parámetros:\n{}".format(best_clf.best_params_['clf']))
    print("Número de variables usadas: {}".format(
        best_clf.best_estimator_['clf'].coef_.shape[1]))
    print("Accuracy en CV: {:.3f}%".format(100.0 * best_clf.best_score_))
    print("Accuracy en training: {:.3f}%".format(
        100.0 * best_clf.score(X_train, y_train)))
    print("Accuracy en test: {:.3f}%".format(
        100.0 * best_clf.score(X_test, y_test)))
    print("Tiempo: {:.3f}s".format(elapsed))

    if show:
        plot_confusion_matrix(best_clf, X_test, y_test)
        plt.show()
        wait()

    if compare:

        # Elegimos un modelo no lineal
        n_trees = 200
        nonlinear_clf = Pipeline([
            ("var", VarianceThreshold(0.1)),
            ("clf", RandomForestClassifier(n_estimators = n_trees, random_state = SEED))])

        # Ajustamos el modelo
        print("\nAjustando modelo no lineal... ", end = "")
        start = default_timer()
        nonlinear_clf.fit(X_train_pre, y_train)
        elapsed = default_timer() - start
        print("Hecho.\n")

        # Mostramos los resultados
        print("--- Clasificador no lineal (RandomForest) ---")
        print("Número de árboles: {}".format(n_trees))
        print("Número de variables usadas: {}".format(X_train_pre.shape[1]))
        print("Accuracy en training: {:.3f}%".format(
            100.0 * nonlinear_clf.score(X_train_pre, y_train)))
        print("Accuracy en test: {:.3f}%".format(
            100.0 * nonlinear_clf.score(X_test_pre, y_test)))
        print("Tiempo: {:.3f}s".format(elapsed))

        # Elegimos un clasificador aleatorio
        dummy_clf = DummyClassifier(strategy = 'stratified', random_state = SEED)

        # Ajustamos el modelo
        print("\nAjustando clasificador aleatorio... ", end = "")
        start = default_timer()
        dummy_clf.fit(X_train_pre, y_train)
        elapsed = default_timer() - start
        print("Hecho.\n")

        # Mostramos los resultados
        print("--- Clasificador aleatorio ---")
        print("Número de variables usadas: {}".format(X_train_pre.shape[1]))
        print("Accuracy en training: {:.3f}%".format(
            100.0 * dummy_clf.score(X_train_pre, y_train)))
        print("Accuracy en test: {:.3f}%".format(
            100.0 * dummy_clf.score(X_test_pre, y_test)))
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
    classification_fit(compare = True, selection_strategy = 0, show = False)

if __name__ == "__main__":
    main()
