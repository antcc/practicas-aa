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
from matplotlib import cm
from timeit import default_timer
from enum import Enum

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import (SelectFromModel, VarianceThreshold, SelectKBest,
    f_classif)
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix

#
# PARÁMETROS GLOBALES
#

SEED = 2020
N_CLASSES = 10
PATH = "datos/"
SAVE_FIGURES = False
IMG_PATH = "img/classification/"

# Estrategias de selección de características
class Selection(Enum):
    PCA = 0
    LASSO = 1
    ANOVA = 2
    NONE = 3

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior."""

    if not SAVE_FIGURES:
        input("\n(Pulsa [Enter] para continuar...)\n")
    plt.close()

def read_data(filename):
    """Lee los datos de dichero y los separa en características y etiquetas."""

    data = np.genfromtxt(filename, delimiter = ",", dtype = np.double)
    return data[:, :-1], data[:, -1]

#
# VISUALIZACIÓN
#

def scatter_plot(X, y, axis, ws = None, labels = None,
                 title = None, figname = "", cmap = cm.tab10):
    """Muestra un scatter plot de puntos etiquetados por clases,
       eventualmente junto a varias rectas de separación.
         - X: matriz de características de la forma [x1, x2].
         - y: vector de etiquetas o clases.
         - axis: nombres de los ejes.
         - ws: lista de vectores 2-dimensionales que representan las rectas
           (se asumen centradas).
         - labels: etiquetas de las rectas.
         - title: título del plot.
         - figname: nombre para guardar la gráfica en fichero."""

    # Establecemos tamaño, colores e información del plot
    plt.figure(figsize = (8, 6))
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    if title is not None:
        plt.title(title)

    # Establecemos los límites del plot
    xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
    ymin, ymax = np.min(X[:, 1]), np.max(X[:, 1])
    scale_x = (xmax - xmin) * 0.01
    scale_y = (ymax - ymin) * 0.01
    plt.xlim(xmin - scale_x, xmax + scale_x)
    plt.ylim(ymin - scale_y, ymax + scale_y)

    # Mostramos scatter plot con leyenda
    scatter = plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap)
    if y is not None:
        legend1 = plt.legend(
            *scatter.legend_elements(),
            title = "Clases",
            loc = "upper right")

    # Pintamos las rectas con leyenda
    if ws is not None:
        # Elegimos los mismos colores que para los puntos
        mask = np.ceil(np.linspace(0, len(cmap.colors) - 1, len(np.unique(y)))).astype(int)
        colors = np.array(cmap.colors)[mask]

        for w, l, c in zip(ws, labels, colors):
            x = np.array([xmin - scale_x, xmax + scale_x])
            plt.plot(x, (-w[0] * x) / w[1], label = l, lw = 2, ls = "--", color = c)

        plt.legend(loc = "lower right")

    # Añadimos leyenda sobre las clases
    if y is not None:
        plt.gca().add_artist(legend1)

    if SAVE_FIGURES:
        plt.savefig(IMG_PATH + figname + ".png")
    else:
        plt.show(block = False)

    wait()

def scatter_pca(X, y_pred):
    """Proyección de las dos primeras componentes principales
       con sus etiquetas predichas.
         - X: matriz de características bidimensionales.
         - y_pred: etiquetas predichas."""

    scatter_plot(
        X[:, [0, 1]],
        y_pred,
        axis = ["Primera componente principal",
            "Segunda componente principal"],
        title = "Proyección de las dos primeras componentes principales",
        figname = "scatter")

def scatter_pca_classes(X, y, ws, classes):
    """Proyección de las dos primeras componentes principales
       con sus etiquetas reales, solo para un subconjunto de clases.
       Se muestran también los clasificadores asociados.
         - X: matriz de características bidimensionales.
         - y: etiquetas reales.
         - ws: coeficientes de los clasificadores.
         - classes: clases a representar."""

    mask = np.in1d(y, classes)
    subset_classes = np.where(mask)[0]
    labels = ["Frontera clase {} vs all".format(i) for i in classes]
    scatter_plot(
        X[subset_classes, :][:, [0, 1]],
        y[subset_classes],
        axis = ["Primera componente principal",
            "Segunda componente principal"],
        ws = ws,
        labels = labels,
        title = "Proyección de las dos primeras componentes principales con clasificadores",
        figname = "scatter_2",
        cmap = cm.tab10)

def confusion_matrix(clf, X, y):
    """Muestra la matriz de confusión de un clasificador en un conjunto de datos.
         - clf: clasificador.
         - X, y: conjunto de datos y etiquetas."""

    fig, ax = plt.subplots(1, 1, figsize = (8, 6))
    disp = plot_confusion_matrix(clf, X, y, cmap = cm.Blues, values_format = 'd', ax = ax)
    disp.ax_.set_title("Matriz de confusión")
    disp.ax_.set_xlabel("Etiqueta predicha")
    disp.ax_.set_ylabel("Etiqueta real")

    if SAVE_FIGURES:
        plt.savefig(IMG_PATH + "confusion.png")
    else:
        plt.show(block = False)
    wait()

def plot_feature_importance(importances, n, pca):
    """Muestra las características más relevantes según el criterio
       inferido por un RandomForest, o bien las primeras componentes principales.
         - importances: vector de relevancia de características.
         - n: número de características a seleccionar.
         - pca: controla si se eligen las primeras componentes principales."""

    if pca:
        indices = range(0, n)
        title = "Importancia de componentes principales"
    else:
        indices = np.argsort(importances)[-n:][::-1]
        title = "Importancia de características"

    # Diagrama de barras para la relevancia
    plt.figure(figsize = (8, 6))
    plt.title(title)
    plt.xlabel("Índice")
    plt.ylabel("Importancia")
    plt.bar(range(n), importances[indices])
    plt.xticks(range(n), indices)
    if SAVE_FIGURES:
        plt.savefig(IMG_PATH + "importance.png")
    else:
        plt.show(block = False)
    wait()

def plot_learning_curve(estimator, X, y, ylim = None, cv = None,
                        n_jobs = None, train_sizes = np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Taken from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    fig, axes = plt.subplots(1, 3, figsize = (16, 6))

    axes[0].set_title("Learning Curves")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Número de ejemplos de entrenamiento")
    axes[0].set_ylabel("Accuracy")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Accuracy en training")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Accuracy en cross-validation")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Número de ejemplos de entrenamiento")
    axes[1].set_ylabel("Tiempos de entrenamiento (s)")
    axes[1].set_title("Escalabilidad del modelo")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Tiempos de entrenamiento (s)")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Desempeño del modelo")

    if SAVE_FIGURES:
        plt.savefig(IMG_PATH + "learning_curve.png")
    else:
        plt.show(block = False)
    wait()

def plot_class_distribution(y_train, y_test):
    """Muestra la distribución de clases en entrenamiento y test."""

    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    plt.suptitle("Distribución de clases", y = 0.96)

    # Diagrama de barras en entrenamiento
    axs[0].bar(np.unique(y_train), np.bincount(y_train.astype(int)),
        color = cm.Set3.colors)
    axs[0].title.set_text("Entrenamiento")
    axs[0].set_xlabel("Clases")
    axs[0].set_ylabel("Número de ejemplos")
    axs[0].set_xticks(range(N_CLASSES))

    # Diagrama de barras en test
    axs[1].bar(np.unique(y_test), np.bincount(y_test.astype(int)),
        color = cm.Set3.colors)
    axs[1].title.set_text("Test")
    axs[1].set_xlabel("Clases")
    axs[1].set_ylabel("Número de ejemplos")
    axs[1].set_xticks(range(N_CLASSES))

    if SAVE_FIGURES:
        plt.savefig(IMG_PATH + "class_distr.png")
    else:
        plt.show(block = False)
    wait()

def plot_corr_matrix(raw, preproc):
    """Muestra la matriz de correlación de un cierto conjunto, antes y
       después del preprocesado.
         - raw: datos antes del preprocesado.
         - preproc: datos tras el preprocesado."""

    fig, axs = plt.subplots(1, 2, figsize = (15, 6))
    fig.suptitle("Matriz de correlación en entrenamiento", y = 0.85)

    # Correlación antes de preprocesar
    with np.errstate(invalid = 'ignore'):
        corr_matrix = np.abs(np.corrcoef(raw, rowvar = False))
    im = axs[0].matshow(corr_matrix, cmap = 'viridis')
    axs[0].title.set_text("Sin preprocesado")

    # Correlación tras preprocesado
    corr_matrix_post = np.abs(np.corrcoef(preproc, rowvar = False))
    axs[1].matshow(corr_matrix_post, cmap = 'viridis')
    axs[1].title.set_text("Con preprocesado")

    fig.colorbar(im, ax = axs.ravel().tolist(), shrink = 0.6)

    if SAVE_FIGURES:
        plt.savefig(IMG_PATH + "correlation.png")
    else:
        plt.show(block = False)
    wait()

#
# PROBLEMA DE CLASIFICACIÓN EN OPTDIGITS
#

def preprocess_pipeline(selection_strategy = Selection.PCA):
    """Construye una lista de transformaciones y parámetros para el
       preprocesamiento de datos, con diferentes estrategias para realizar
       selección de variables."""

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
    start = default_timer()
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
        plot_class_distribution(y_train, y_test)

        # Mostramos matriz de correlación de training antes y después de preprocesado
        plot_corr_matrix(X_train_raw, X_train)

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
    print("Accuracy en training: {:.3f}%".format(
        100.0 * best_clf.score(X_train, y_train)))
    print("Accuracy en test: {:.3f}%".format(
        100.0 * best_clf.score(X_test, y_test)))
    print("Tiempo: {:.3f}s".format(elapsed))

    # Gráficas y visualización
    if show > 0:
        print("\nMostrando gráficas sobre entrenamiento y predicción...")

        # Matriz de confusión
        confusion_matrix(best_clf, X_test, y_test)

        if show > 1 and selection_strategy is not Selection.LASSO:
            # Curva de aprendizaje
            print("Calculando curva de aprendizaje...")
            plot_learning_curve(best_clf, X_train, y_train, n_jobs = -1)

        # Importancia de características
        pipe = Pipeline(pipe_lst + [("clf", RandomForestClassifier(random_state = SEED))])
        pipe.fit(X_train, y_train)
        importances = pipe['clf'].feature_importances_
        plot_feature_importance(importances, 10, selection_strategy == Selection.PCA)

        # Visualización de componentes principales
        if selection_strategy == Selection.PCA:
            # Predicciones para el conjunto de test
            y_pred = best_clf.predict(X_test)

            # Proyección de las dos primeras componentes principales
            # con etiquetas predichas
            scatter_pca(X_test, y_pred)

            # Seleccionamos dos clases concretas y mostramos también los clasificadores,
            # frente a las etiquetas reales
            classes = [1, 2, 3]
            coef = best_clf.best_estimator_['clf'].coef_
            ws = [[coef[i, 0], coef[i, 1]] for i in classes]
            scatter_pca_classes(X_test, y_test, ws, classes)

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
        print("Accuracy en training: {:.3f}%".format(
            100.0 * nonlinear_clf.score(X_train_raw, y_train)))
        print("Accuracy en test: {:.3f}%".format(
            100.0 * nonlinear_clf.score(X_test_raw, y_test)))
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
        print("Accuracy en training: {:.3f}%".format(
            100.0 * dummy_clf.score(X_train_raw, y_train)))
        print("Accuracy en test: {:.3f}%".format(
            100.0 * dummy_clf.score(X_test_raw, y_test)))
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
