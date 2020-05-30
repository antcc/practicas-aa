#!/usr/bin/env python
# coding: utf-8
# uso: ./p3_regression.py

"""
Aprendizaje Automático. Curso 2019/20.
Práctica 3: Ajuste de modelos lineales. Regresión
Intentamos conseguir el mejor ajuste con modelos lineales para un
problema de regresión.

Base de datos: Communities and Crimes
http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime

Antonio Coín Castro. Grupo 3.
"""

#
# LIBRERÍAS
#

import numpy as np
from timeit import default_timer
from pandas import read_csv

from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso, SGDRegressor, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from p3_visualization import (plot_feature_importance, plot_learning_curve,
    plot_corr_matrix, plot_features, plot_hist_dependent, plot_scatter_pca_reg,
    plot_residues_error, wait)

#
# PARÁMETROS GLOBALES
#

SEED = 2020
PATH = "datos/"
SAVE_FIGURES = False
IMG_PATH = "img/regression/"

#
# PROBLEMA DE REGRESIÓN
#

def print_evaluation_metrics(reg, X_train, X_test, y_train, y_test):
    """Imprime la evaluación de resultados en training y test de un regresor."""

    for name, X, y in [("training", X_train, y_train), ("test", X_test, y_test)]:
        y_pred = reg.predict(X)
        print("RMSE en {}: {:.3f}".format(
            name, (np.sqrt(mean_squared_error(y, y_pred)))))
        print("R2 en {}: {:.3f}".format(
            name, r2_score(y, y_pred)))

def load_and_split_data(filename, test_size = 0.2):
    """Carga los datos de fichero, trata los valores perdidos, y realiza una
       división training/test en la proporción indicada."""

    # Cargamos los datos
    df = read_csv(filename, header = None, na_values = '?')

    # Eliminamos las 5 primeras columnas (no son predictores)
    df.drop(df.index[:5], axis = 1, inplace = True)

    # Eliminamos columnas con más de la mitad de valores perdidos
    df = df[df.columns[df.isna().mean() <= 0.5]]

    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    # Realizamos división en training y test
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size = test_size, random_state = SEED)

    # Imputamos el resto de valores perdidos con la mediana
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.fit_transform(X_test)

    return X_train, X_test, y_train, y_test

def preprocess_pipeline():
    """Construye una lista de transformaciones para el
       preprocesamiento de datos, con transformaciones polinómicas
       de grado 2 y selección de características."""

    preproc = [
        ("selection", PCA(0.80)),
        ("standardize", StandardScaler()),
        ("poly", PolynomialFeatures(2)),
        ("var", VarianceThreshold()),
        ("standardize2", StandardScaler())]

    return preproc

def regression_fit(compare = False, show = 0):
    """Ajuste de un modelo lineal para resolver un problema de regresión.
       Opcionalmente se puede ajustar también un modelo no lineal (RandomForest)
       y un regresor aleatorio para comparar el rendimiento.
         - compare: controla si se realizan comparaciones con otros regresores.
         - show: controla si se muestran gráficas informativas, a varios niveles
             * 0: No se muestran.
             * 1: Se muestran las que no consumen demasiado tiempo.
             * >=2: Se muestran todas."""

    # Cargamos los datos de entrenamiento y test
    print("Cargando datos de entrenamiento y test... ", end = "", flush = True)
    X_train, X_test, y_train, y_test = \
        load_and_split_data(PATH + "communities.data", test_size = 0.2)
    print("Hecho.")

    if show > 0:
        print("Mostrando gráficas de inspección de los datos...")

        # Visualizamos las variables más relevantes
        names = ["PctYoungKids2Par", "MalePctNevMarr"]
        features = [50, 44]
        plot_features(features, names, X_train, y_train, SAVE_FIGURES, IMG_PATH)

        # Mostramos histograma de la variable dependiente
        plot_hist_dependent(y_train, SAVE_FIGURES, IMG_PATH)

    # Creamos un pipeline de preprocesado
    preproc = preprocess_pipeline()
    preproc_pipe = Pipeline(preproc)

    # Obtenemos los datos preprocesados por si los necesitamos
    X_train_pre = preproc_pipe.fit_transform(X_train, y_train)
    X_test_pre = preproc_pipe.transform(X_test)

    # Construimos un pipeline de preprocesado + regresión
    # (el regresor que ponemos puede ser cualquiera, es un 'placeholder')
    pipe = Pipeline(preproc + [("reg", LinearRegression())])

    if show > 0:
        print("Mostrando gráficas sobre preprocesado y características...")

        # Mostramos matriz de correlación de training antes y después de preprocesado
        plot_corr_matrix(X_train, X_train_pre, SAVE_FIGURES, IMG_PATH)

        if show > 1:
            # Importancia de características
            pipe = Pipeline(preproc + [("reg", RandomForestRegressor(random_state = SEED))])
            pipe.fit(X_train, y_train)
            importances = pipe['reg'].feature_importances_
            plot_feature_importance(importances, 10, True, SAVE_FIGURES, IMG_PATH)

    # Elegimos los modelos lineales y sus parámetros para CV
    max_iter = 2000
    search_space = [
        {"reg": [SGDRegressor(penalty = 'l2',
                              max_iter = max_iter,
                              random_state = SEED)],
         "reg__alpha": np.logspace(-4, 4, 10)},
        {"reg": [Ridge(max_iter = max_iter)],
         "reg__alpha": np.logspace(-4, 4, 10)},
        {"reg": [Lasso(random_state = SEED,
                       max_iter = max_iter)],
         "reg__alpha": np.logspace(-4, 4, 10)}]

    # Buscamos los mejores parámetros por CV
    print("Realizando selección de modelos lineales... ", end = "", flush = True)
    start = default_timer()
    best_reg = GridSearchCV(pipe, search_space, scoring = 'neg_mean_squared_error',
        cv = 5, n_jobs = -1)
    best_reg.fit(X_train, y_train)
    elapsed = default_timer() - start
    print("Hecho.\n")

    # Mostramos los resultados
    print("--- Mejor regresor lineal ---")
    print("Parámetros:\n{}".format(best_reg.best_params_['reg']))
    print("Número de variables usadas: {}".format(
        best_reg.best_estimator_['reg'].coef_.shape[0]))
    print("RMSE en CV: {:.3f}".format(np.sqrt(-best_reg.best_score_)))
    print_evaluation_metrics(best_reg, X_train, X_test, y_train, y_test)
    print("Tiempo: {:.3f}s".format(elapsed))

    # Gráficas y visualización
    if show > 0:
        wait()
        print("Mostrando gráficas sobre entrenamiento y predicción...")

        # Visualización de residuos y error de predicción
        y_pred = best_reg.predict(X_test)
        plot_residues_error(y_test, y_pred, SAVE_FIGURES, IMG_PATH)

        # Visualización de componentes principales
        m = best_reg.best_estimator_['reg'].coef_[0]
        b = best_reg.best_estimator_['reg'].intercept_
        plot_scatter_pca_reg(X_test_pre[:, 0], y_test, m, b, SAVE_FIGURES, IMG_PATH)

        if show > 1:
            # Curva de aprendizaje
            print("Calculando curva de aprendizaje...")
            plot_learning_curve(best_reg, X_train, y_train, n_jobs = -1,
                cv = 5, scoring = 'neg_mean_squared_error',
                save_figures = SAVE_FIGURES, img_path = IMG_PATH)
            elapsed = default_timer() - start
            print("Tiempo: {:.3f}s".format(elapsed))

    # Comparación con modelos no lineales
    if compare:
        # Elegimos un modelo no lineal
        n_trees = 200
        nonlinear_reg = Pipeline([
            ("var", VarianceThreshold()),
            ("reg", RandomForestRegressor(n_estimators = n_trees,
                max_depth = 10, random_state = SEED))])

        # Ajustamos el modelo
        print("\nAjustando modelo no lineal... ", end = "", flush = True)
        start = default_timer()
        nonlinear_reg.fit(X_train, y_train)
        elapsed = default_timer() - start
        print("Hecho.\n")

        # Mostramos los resultados
        print("--- Regresor no lineal (RandomForest) ---")
        print("Número de árboles: {}".format(n_trees))
        print("Número de variables usadas: {}".format(X_train.shape[1]))
        print_evaluation_metrics(nonlinear_reg, X_train, X_test, y_train, y_test)
        print("Tiempo: {:.3f}s".format(elapsed))

        # Elegimos un regresor aleatorio
        dummy_reg = DummyRegressor(strategy = 'mean')

        # Ajustamos el modelo
        print("\nAjustando regresor aleatorio... ", end = "", flush = True)
        start = default_timer()
        dummy_reg.fit(X_train, y_train)
        elapsed = default_timer() - start
        print("Hecho.\n")

        # Mostramos los resultados
        print("--- Regresor aleatorio ---")
        print("Número de variables usadas: {}".format(X_train.shape[1]))
        print_evaluation_metrics(dummy_reg, X_train, X_test, y_train, y_test)
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
    print("--- PARTE 2: REGRESIÓN ---")

    start = default_timer()
    regression_fit(compare = True, show = 1)
    elapsed = default_timer() - start
    print("\nTiempo total de ejecución: {:.3f}s".format(elapsed))

if __name__ == "__main__":
    main()
