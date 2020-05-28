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
from timeit import default_timer
from enum import Enum

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import (SelectFromModel, VarianceThreshold, SelectKBest,
    f_classif)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from p3_visualization import (scatter_pca, scatter_pca_classes, confusion_matrix,
    plot_feature_importance, plot_learning_curve, plot_class_distribution, plot_corr_matrix)

#
# PARÁMETROS GLOBALES
#

SEED = 2020
N_CLASSES = 10
PATH = "datos/"
SAVE_FIGURES = False
IMG_PATH = "img/classification/"

#
# PROBLEMA DE CLASIFICACIÓN
#

class Selection(Enum):
    """Estrategia de selección de características."""
    PCA = 0
    LASSO = 1
    ANOVA = 2
    NONE = 3

def print_evaluation_metrics(clf, X_train, X_test, y_train, y_test):
    """Imprime la evaluación de resultados en training y test de un clasificador."""

    for name, X, y in [("training", X_train, y_train), ("test", X_test, y_test)]:
        print("Accuracy en {}: {:.3f}%".format(
            name, 100.0 * clf.score(X, y)))

def read_data(filename):
    """Lee los datos de dichero y los separa en características y etiquetas."""

    data = np.genfromtxt(filename, delimiter = ",", dtype = np.double)
    return data[:, :-1], data[:, -1]

def preprocess_pipeline(selection_strategy = Selection.PCA):
    """Construye una lista de transformaciones para el
       preprocesamiento de datos, con transformaciones polinómicas
       de grado 2 y diferentes estrategias para realizar selección
       de variables."""

    # Reducción de dimensionalidad
    if selection_strategy == Selection.PCA:
        preproc = [
            ("selection", PCA(0.95)),
            ("standardize", StandardScaler()),
            ("poly", PolynomialFeatures(2)),
            ("var", VarianceThreshold(0.1)),
            ("standardize2", StandardScaler())]

    # Selección de variables
    else:
        preproc = [
            ("standardize", StandardScaler()),
            ("poly", PolynomialFeatures(2)),
            ("var", VarianceThreshold(0.1)),
            ("standardize2", StandardScaler())]

    return preproc

def classification_fit(compare = False, selection_strategy = Selection.PCA, show = 0):
    """Ajuste de un modelo lineal para resolver un problema de clasificación.
       Opcionalmente se puede ajustar también un modelo no lineal (RandomForest)
       y un clasificador aleatorio para comparar el rendimiento.
         - compare: controla si se realizan comparaciones con otros clasificadores.
         - selection_strategy: estrategia de selección de características
             * 0: Mediante PCA.
             * 1: Mediante regresión L1 eliminando los coeficientes que vayan a 0.
             * 2: Mediante f-test (ANOVA).
           Cualquier otro número hace que no se realice selección de características.
         - show: controla si se muestran gráficas informativas, a varios niveles
             * 0: No se muestran.
             * 1: Se muestran las que no consumen demasiado tiempo.
             * >=2: Se muestran todas."""

    # Cargamos los datos de entrenamiento y test
    print("Cargando datos de entrenamiento y test... ", end = "")
    X_train_raw, y_train = read_data(PATH + "optdigits.tra")
    X_test_raw, y_test = read_data(PATH + "optdigits.tes")
    print("Hecho.")

    # Preprocesamos los datos
    print("Preprocesando datos... ", end = "")
    preproc = preprocess_pipeline(selection_strategy)
    preproc_pipe = Pipeline(preproc)
    X_train = preproc_pipe.fit_transform(X_train_raw, y_train)
    X_test = preproc_pipe.transform(X_test_raw)
    print("Hecho.")

    if show > 0:
        print("\nMostrando gráficas sobre preprocesado...")

        # Mostramos distribución de clases en training y test
        plot_class_distribution(y_train, y_test, N_CLASSES, SAVE_FIGURES, IMG_PATH)

        # Mostramos matriz de correlación de training antes y después de preprocesado
        plot_corr_matrix(X_train_raw, X_train, SAVE_FIGURES, IMG_PATH)

    # Construimos un pipeline de selección + clasificación
    pipe_lst = []
    if selection_strategy == Selection.LASSO:
        pipe_lst += [("var2", VarianceThreshold()),
                     ("selection", SelectFromModel(Lasso(alpha = 0.005), threshold = 0.01))]
    elif selection_strategy == Selection.ANOVA:
        pipe_lst += [("var2", VarianceThreshold()),
                     ("selection", SelectKBest(f_classif, k = X_train.shape[1] // 3))]
    pipe = Pipeline(pipe_lst + [("clf", LogisticRegression())])

    # Elegimos los modelos lineales y sus parámetros para CV
    search_space = [
        {"clf": [LogisticRegression(multi_class = 'ovr', penalty = 'l2', max_iter = 500)],
         "clf__C": np.logspace(-4, 4, 3)},
        {"clf": [SGDClassifier(loss = 'hinge', random_state = SEED)],
         "clf__alpha": np.logspace(-4, 4, 3)},
        {"clf": [RidgeClassifier(random_state = SEED)],
         "clf__alpha": np.logspace(-4, 4, 3)}]

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
    print_evaluation_metrics(best_clf, X_train, X_test, y_train, y_test)
    print("Tiempo: {:.3f}s".format(elapsed))

    # Gráficas y visualización
    if show > 0:
        print("\nMostrando gráficas sobre entrenamiento y predicción...")

        # Matriz de confusión
        confusion_matrix(best_clf, X_test, y_test, SAVE_FIGURES, IMG_PATH)

        if show > 1 and selection_strategy is not Selection.LASSO:
            # Curva de aprendizaje
            print("Calculando curva de aprendizaje...")
            plot_learning_curve(best_clf, X_train, y_train, n_jobs = -1,
                save_figures = SAVE_FIGURES, img_path = IMG_PATH)

        # Importancia de características
        pipe = Pipeline(pipe_lst + [("clf", RandomForestClassifier(random_state = SEED))])
        pipe.fit(X_train, y_train)
        importances = pipe['clf'].feature_importances_
        plot_feature_importance(importances, 10, selection_strategy == Selection.PCA,
            SAVE_FIGURES, IMG_PATH)

        # Visualización de componentes principales
        if selection_strategy == Selection.PCA:
            # Predicciones para el conjunto de test
            y_pred = best_clf.predict(X_test)

            # Proyección de las dos primeras componentes principales
            # con etiquetas predichas
            scatter_pca(X_test, y_pred, SAVE_FIGURES, IMG_PATH)

            # Seleccionamos dos clases concretas y mostramos también los clasificadores,
            # frente a las etiquetas reales
            classes = [1, 2, 3]
            coef = best_clf.best_estimator_['clf'].coef_
            ws = [[coef[i, 0], coef[i, 1]] for i in classes]
            scatter_pca_classes(X_test, y_test, ws, classes, SAVE_FIGURES, IMG_PATH)

    # Comparación con modelos no lineales
    if compare:
        # Elegimos un modelo no lineal
        n_trees = 200
        nonlinear_clf = Pipeline([
            ("var", VarianceThreshold(0.1)),
            ("clf", RandomForestClassifier(n_estimators = n_trees, random_state = SEED))])

        # Ajustamos el modelo
        print("\nAjustando modelo no lineal... ", end = "")
        start = default_timer()
        nonlinear_clf.fit(X_train_raw, y_train)
        elapsed = default_timer() - start
        print("Hecho.\n")

        # Mostramos los resultados
        print("--- Clasificador no lineal (RandomForest) ---")
        print("Número de árboles: {}".format(n_trees))
        print("Número de variables usadas: {}".format(X_train_raw.shape[1]))
        print_evaluation_metrics(nonlinear_clf, X_train_raw, X_test_raw, y_train, y_test)
        print("Tiempo: {:.3f}s".format(elapsed))

        # Elegimos un clasificador aleatorio
        dummy_clf = DummyClassifier(strategy = 'stratified', random_state = SEED)

        # Ajustamos el modelo
        print("\nAjustando clasificador aleatorio... ", end = "")
        start = default_timer()
        dummy_clf.fit(X_train_raw, y_train)
        elapsed = default_timer() - start
        print("Hecho.\n")

        # Mostramos los resultados
        print("--- Clasificador aleatorio ---")
        print("Número de variables usadas: {}".format(X_train_raw.shape[1]))
        print_evaluation_metrics(dummy_clf, X_train_raw, X_test_raw, y_train, y_test)
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

    start = default_timer()
    classification_fit(compare = True, selection_strategy = Selection.PCA, show = 1)
    elapsed = default_timer() - start
    print("\nTiempo total de ejecución: {:.3f}s".format(elapsed))

if __name__ == "__main__":
    main()
