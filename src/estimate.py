from random import random
from typing import Callable

def mean(
  tol: float,
  X: Callable[..., float],
  *X_args: ...
) -> float:
  """Media muestral.

  Parámetros
  ----------
  tol : float
    Máxima tolerancia deseada para el estimador de la varianza muestral.
  X : (...) -> float
    Función generadora de una variable aleatoria.
  *X_args : (...)
    Argumentos con los que llamar a `X`.

  Retorna
  -------
  float
    Media de una muestra cuyo tamaño es tal que el estimador de su varianza
    muestral no supera la tolerancia deseada.
  """
  n, E_prev, E, S_sqr = 0, 0, 0, 0
  tol_sqr = tol**2

  while n < 100 or n < S_sqr / tol_sqr:
    n += 1
    E_prev = E
    E += (X(*X_args) - E_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (E - E_prev)**2

  return E

def mean_conf_int(
  z_alfa_2 : float,
  L : float,
  X : Callable[..., float],
  *X_args : ...
) -> float:
  """Media muestral por intervalo de confianza.

  Parámetros
  ----------
  z_alfa_2 : float
    Valor crítico asociado al intervalo de confianza.
  L : float
    Máxima tolerancia deseada para la longitud del intervalo de confianza.
  X : (...) -> float
    Función generadora de una variable aleatoria.
  *X_args : (...)
    Argumentos con los que llamar a `X`.

  Retorna
  -------
  float
    Media de una muestra que pertenece a un intervalo de confianza
    `(1-alfa)*100%` cuyo tamaño no supera 'L'.
  """
  return mean(L / (2 * z_alfa_2), X, *X_args)

def monte_carlo(
  tol: float,
  h: Callable[[float], float]
) -> float:
  """Monte carlo para la integral desde 0 a 1 de `h(x) dx`.

  Parámetros
  ----------
  tol : float
    Máxima tolerancia deseada para el estimador de la varianza muestral.
  h :
    Función a integrar.

  Retorna
  -------
  float
    Integral desde 0 a 1 de `h(x) dx`
  """
  return mean(tol, h, 1 - random())