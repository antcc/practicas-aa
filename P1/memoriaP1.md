---
title: Visión por Computador
subtitle: "Práctica 3: Detección de puntos relevantes y construcción de panoramas"
author: Antonio Coín Castro
date: Curso 2019-20
geometry: margin = 1.2in
documentclass: scrartcl
colorlinks: true
urlcolor: Magenta
lang: es
header-includes:
  - \usepackage{amsmath}
  - \usepackage{graphicx}
  - \usepackage{subcaption}
---

# Estructura del código

Se han desarrollado una serie de funciones genéricas de representación de imágenes, extraídas en su mayoría de prácticas anteriores (aunque con ligeras modificaciones). Hay una serie de consideraciones a tener en cuenta:

- En general, todas las imágenes se convierten a números reales en el momento en que se leen, y solo se normalizan a $[0,1]$ cuando se vayan a pintar.
- Tras mostrar una ventana de imagen, el programa quedará a la espera de una pulsación de tecla para continuar con su ejecución (incluso si se cierra manualmente la ventana del *plot*).
- Hay una serie de parámetros globales (editables) al inicio del programa para modificar ciertos comportamientos de algunas funciones.
- Las imágenes deben estar en la ruta relativa `imagenes/`.
- Todas las funciones están comentadas y se explica el significado de los parámetros.

El programa desarrollado va ejecutando desde una función `main` los distintos apartados de la práctica uno por uno, llamando en cada caso a las funciones que sean necesarias para mostrar ejemplos de funcionamiento.

# Ejercicio 1:

se incluye todo en `gd` para que sirva para todos.

# Ejercicio 2:

comparar con Pseudoinversa en el experimento

# Bibliografía

**[1]** [Multi-Image Matching using Multi-Scale Oriented Patches](http://matthewalunbrown.com/papers/cvpr05.pdf)

**[2]** [Distinctive Image Features from Scale-Invariant Keypoints](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
