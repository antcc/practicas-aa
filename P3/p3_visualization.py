# coding: utf-8

"""
Aprendizaje Automático. Curso 2019/20.
Práctica 3: Ajuste de modelos lineales. Visualización.
Colección de funciones para visualización de gráficas para
los dos problemas de esta práctica.

Todas las funciones tienen parámetros 'save_figures' y 'img_path'
que permiten guardar las imágenes generadas en disco en vez de
mostrarlas.

Antonio Coín Castro. Grupo 3.
"""

#
# LIBRERÍAS
#

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression

#
# COMUNES
#

def wait(save_figures = False):
    """Introduce una espera hasta que se pulse el intro.
       Limpia el plot anterior."""

    if not save_figures:
        input("\n(Pulsa [Enter] para continuar...)\n")
    plt.close()

def scatter_plot(X, y, axis, ws = None, labels = None, title = None,
                 figname = "", cmap = cm.tab10, save_figures = False, img_path = ""):
    """Muestra un scatter plot de puntos (opcionalmente) etiquetados por clases,
       eventualmente junto a varias rectas de separación.
         - X: matriz de características de la forma [x1, x2].
         - y: vector de etiquetas o clases. Puede ser None.
         - axis: nombres de los ejes.
         - ws: lista de vectores 2-dimensionales que representan las rectas
           (se asumen centradas).
         - labels: etiquetas de las rectas.
         - title: título del plot.
         - figname: nombre para guardar la gráfica en fichero.
         - cmap: mapa de colores."""

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

    if save_figures:
        plt.savefig(img_path + figname + ".png")
    else:
        plt.show(block = False)

    wait(save_figures)

def plot_corr_matrix(raw, preproc, save_figures = False, img_path = ""):
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

    if save_figures:
        plt.savefig(img_path + "correlation.png")
    else:
        plt.show(block = False)
    wait(save_figures)

def plot_feature_importance(importances, n, pca, save_figures = False, img_path = ""):
    """Muestra las características más relevantes obtenidas según algún
       criterio, o bien las primeras componentes principales.
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
    if save_figures:
        plt.savefig(img_path + "importance.png")
    else:
        plt.show(block = False)
    wait(save_figures)

def plot_learning_curve(estimator, X, y, scoring, ylim = None, cv = None,
                        n_jobs = None, train_sizes = np.linspace(.1, 1.0, 5),
                        save_figures = False, img_path = ""):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Adapted from:
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

    scoring : A str (see model evaluation documentation) or a scorer callable
        object / function with signature scorer(estimator, X, y)

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

    if scoring == 'accuracy':
        score_name = "Accuracy"
    elif scoring == 'neg_mean_squared_error':
        score_name = "RMSE"
    else:
        score_name = scoring

    axes[0].set_title("Learning Curves")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Número de ejemplos de entrenamiento")
    axes[0].set_ylabel(score_name)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)

    if scoring == 'neg_mean_squared_error':
        train_scores = np.sqrt(-train_scores)
        test_scores = np.sqrt(-test_scores)

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
                 label=score_name + " en training")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label=score_name + " en cross-validation")
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
    axes[2].set_ylabel(score_name)
    axes[2].set_title("Desempeño del modelo")

    if save_figures:
        plt.savefig(img_path + "learning_curve.png")
    else:
        plt.show(block = False)
    wait(save_figures)

#
# CLASIFICACIÓN
#

def scatter_pca(X, y_pred, save_figures = False, img_path = ""):
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
        figname = "scatter",
        save_figures = save_figures,
        img_path = img_path)

def scatter_pca_classes(X, y, ws, classes, save_figures = False, img_path = ""):
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
        cmap = cm.tab10,
        save_figures = save_figures,
        img_path = img_path)

def confusion_matrix(clf, X, y, save_figures = False, img_path = ""):
    """Muestra la matriz de confusión de un clasificador en un conjunto de datos.
         - clf: clasificador.
         - X, y: conjunto de datos y etiquetas."""

    fig, ax = plt.subplots(1, 1, figsize = (8, 6))
    disp = plot_confusion_matrix(clf, X, y, cmap = cm.Blues, values_format = 'd', ax = ax)
    disp.ax_.set_title("Matriz de confusión")
    disp.ax_.set_xlabel("Etiqueta predicha")
    disp.ax_.set_ylabel("Etiqueta real")

    if save_figures:
        plt.savefig(img_path + "confusion.png")
    else:
        plt.show(block = False)
    wait(save_figures)

def plot_class_distribution(y_train, y_test, n_classes, save_figures = False, img_path = ""):
    """Muestra la distribución de clases en entrenamiento y test."""

    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    plt.suptitle("Distribución de clases", y = 0.96)

    # Diagrama de barras en entrenamiento
    axs[0].bar(np.unique(y_train), np.bincount(y_train.astype(int)),
        color = cm.Set3.colors)
    axs[0].title.set_text("Entrenamiento")
    axs[0].set_xlabel("Clases")
    axs[0].set_ylabel("Número de ejemplos")
    axs[0].set_xticks(range(n_classes))

    # Diagrama de barras en test
    axs[1].bar(np.unique(y_test), np.bincount(y_test.astype(int)),
        color = cm.Set3.colors)
    axs[1].title.set_text("Test")
    axs[1].set_xlabel("Clases")
    axs[1].set_ylabel("Número de ejemplos")
    axs[1].set_xticks(range(n_classes))

    if save_figures:
        plt.savefig(img_path + "class_distr.png")
    else:
        plt.show(block = False)
    wait(save_figures)

#
# REGRESIÓN
#

def plot_features(features, names, X, y, save_figures = False, img_path = ""):
    """Muestra algunos predictores frente a la variable dependiente. Además
       ajusta una recta de regresión.
         - features: vector numérico de características.
         - names: nombres de las características.
         - X: matriz de características.
         - y: vector con la variable a predecir."""

    fig, axs = plt.subplots(1, len(features), figsize = (12, 6))
    fig.suptitle("Visualización de algunas características", y = 0.95)

    reg = LinearRegression()

    for (i, f), name in zip(enumerate(features), names):
        # Mostramos un scatter
        x = X[:, f].reshape(X.shape[0], 1)
        axs[i].scatter(x, y, facecolors = 'none', edgecolors = 'tab:blue', marker = '.')
        axs[i].set_xlabel("{} (nº {})".format(name, f))
        axs[i].set_ylabel("ViolentCrimesPerPop")

        # Ajustamos una recta y la mostramos
        reg.fit(x, y)
        m = reg.coef_[0]
        b = reg.intercept_
        xx = np.array([x.min() - 0.1, x.max() + 0.1])
        axs[i].plot(xx, m * xx + b, color = 'k', ls = "--", lw = 2)

    if save_figures:
        plt.savefig(img_path + "features_y.png")
    else:
        plt.show(block = False)
    wait(save_figures)

def plot_hist_dependent(y, save_figures = False, img_path = ""):
    """Muestra un histograma de la distribución de la variable dependiente."""

    plt.figure(figsize = (8, 6))
    plt.xlabel("ViolentCrimesPerPop")
    plt.ylabel("Frecuencia")
    plt.title("Histograma de ViolentCrimesPerPop")
    plt.hist(y, edgecolor = 'k', bins = 15)

    if save_figures:
        plt.savefig(img_path + "hist_y.png")
    else:
        plt.show(block = False)
    wait(save_figures)

def plot_scatter_pca_reg(x, y, m, b, save_figures = False, img_path = ""):
    """Proyecta la primera componente principal frente a la variable dependiente,
       junto al ajuste obtenido.
         - x: primera componente principal.
         - y: vector con la variable a predecir
         - m: pendiente de la recta de ajuste.
         - b: punto de corte con el eje Y de la recta de ajuste."""

    plt.figure(figsize = (8, 6))
    plt.xlabel("Primera componente principal")
    plt.ylabel("ViolentCrimesPerPop")
    plt.title("Proyección de la primera componente principal y recta de regresión")

    # Mostramos un scatter
    plt.scatter(x, y, facecolors = 'none', edgecolors = 'tab:blue', marker = '.')

    # Mostramos la recta de ajuste
    xx = np.array([x.min() - 0.1, x.max() + 0.1])
    plt.plot(xx, m * xx + b, color = 'k', ls = "--", lw = 2)

    if save_figures:
        plt.savefig(img_path + "scatter_reg.png")
    else:
        plt.show(block = False)
    wait(save_figures)

def plot_residues_error(y_true, y_pred, save_figures = False, img_path = ""):
    """Muestra un gráfico de los residuos tras una regresión lineal,
       y otro gráfico del error de predicción."""

    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    fig.suptitle("Algunos gráficos sobre el error en el ajuste", y = 0.95)

    reg = LinearRegression()

    # Gráfico de residuos
    x = y_pred.reshape(-1, 1)
    y = y_true - y_pred
    axs[0].scatter(x, y, facecolors = 'none', edgecolors = 'tab:blue', marker = '.')
    axs[0].set_xlabel("ViolentCrimesPerPop")
    axs[0].set_ylabel("Residuos")
    axs[0].set_title("Diagrama de residuos")

    # Ajustamos una recta y la mostramos
    reg.fit(x, y)
    m = reg.coef_[0]
    b = reg.intercept_
    xx = np.array([x.min() - 0.1, x.max() + 0.1])
    axs[0].plot(xx, m * xx + b, color = 'k', ls = "--", lw = 2)

    # Gráfico de error de predicción
    x = y_true.reshape(-1, 1)
    y = y_pred
    axs[1].scatter(x, y, facecolors = 'none', edgecolors = 'tab:blue', marker = '.')
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("y_pred")
    axs[1].set_title("Error de predicción")

    # Ajustamos una recta y la mostramos junto a la recta y = x
    reg.fit(x, y)
    m = reg.coef_[0]
    b = reg.intercept_
    xx = np.array([x.min() - 0.1, x.max() + 0.1])
    axs[1].plot(xx, m * xx + b, color = 'k', ls = "--", lw = 2, label = "Mejor ajuste")
    axs[1].plot(xx, xx, color = 'gray', ls = "--", lw = 2, label = "Identidad")
    axs[1].legend()

    if save_figures:
        plt.savefig(img_path + "residues_error.png")
    else:
        plt.show(block = False)
    wait(save_figures)
