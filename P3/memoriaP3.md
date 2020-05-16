---
title: Aprendizaje automático
subtitle: "Práctica 3: Ajuste de modelos lineales "
author: Antonio Coín Castro
date: Curso 2019-20
geometry: margin = 1.2in
documentclass: scrartcl
colorlinks: true
urlcolor: Magenta
header-includes:
  - \usepackage{amsmath}
  - \usepackage{graphicx}
  - \usepackage{subcaption}
  - \usepackage[spanish]{babel}
---

\decimalpoint

# Notas

## Preprocesado de datos

- Valores perdidos

- Clases balanceadas?

Centramos y normalizamos en escala. Se hacen **solo** en el conjunto de entrenamiento, se guardan los parámetros, y después se realizan las mismas en el test justo antes de predecir.

### Reducción de la dimensión

*The curse of dimensionality is a general observation that statistical tasks get exponentially harder as the dimensions increase.* Selección de características.

PCA vs hacer regresión con error L1 y ver qué pesos van a 0.

Ahora podemos hacer *whitening* para tratar todas las dimensiones de la misma forma y disminuir correlaciones.

## Selección de la clase de funciones

O dejarlo como está o usar polinomios de grado 2.
