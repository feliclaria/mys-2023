from random import random, gauss, gammavariate
from typing import *
import discrete as disc
import math

def exponential(lambd: float) -> float:
  """Distribución exponencial.

  Parametros
  ----------
  lambd : float
    `1.0` dividido por la media deseada. Debe ser distinto de cero.

  Retorna
  -------
  float
    Valor entre 0 e infinito si `lambd > 0`, y entre infinito negativo y 0 si `lambd < 0`.
  """
  return - math.log(1 - random()) / lambd


_NORMAL_MAGIC_CONST_A = 2 * math.exp(-1)
_NORMAL_MAGIC_CONST_B = 4 * math.exp(-0.5) / math.sqrt(2)

def normal(mu: float, sigma: float) -> float:
  """Distribución normal.

  Parametros
  ----------
  mu : float
    Media deseada.
  sigma : float
    Desviación estándar deseada.

  Retorna
  -------
  float
    Valor entre -infinito e infinito.
  """
  u, v = random(), 1 - random()
  X = (u - 0.5) / v

  while _NORMAL_MAGIC_CONST_A * X**2 > -math.log(v):
    u, v = random(), 1 - random()
    X = (u - 0.5) / v

  return  _NORMAL_MAGIC_CONST_B * X * sigma + mu


def rt(df: int) -> float:
  """Distribución t-student.

  Parametros
  ----------
  df : int
    Grados de libertad.

  Retorna
  -------
  result : float
  """
  x = gauss(0.0, 1.0)
  y = 2.0 * gammavariate(0.5 * df, 2.0)
  return x / (math.sqrt(y/df))


def composition_method(
  gens: List[Callable[[], float]],
  probs: List[float]
) -> float:
  """Método de composición.

  Método de composición para la generación de una variable aleatoria continua
  `X` con función de distribución acumulada dada por `F(x) = p_1 * F_1(x) +
  ... + p_n * F_n(x)`, con cada `F_i` función de distribución acumulada de
  una variable aleatoria continua `X_i`, y cada `p_i` constante positiva tal
  que `p_1 + ... + p_n = 1`.

  Parametros
  ----------
  gens : List[() -> float]
    Funciones generadoras de `X_1, ..., X_n`.
  probs : List[float]
    Probabilidades `p_1, ..., p_n`.

  Retorna
  -------
  float
    Un valor en el rango de `X`.
  """
  n = len(gens)
  i = disc.inverse_trans_arr(probs, list(range(n)))
  X_i = gens[i]
  return X_i()


def accept_reject_method(
  Y_gen: Callable[[], float],
  f: Callable[[float], float],
  g: Callable[[float], float],
  c: float
) -> float:
  """Método de aceptación y rechazo.

  Método de aceptación y rechazo para la generación de una variable aleatoria
  continua `X`.

  Parámetros
  ----------
  Y_gen : () -> float
    Función generadora de una variable aleatoria continua `Y`.
  f : (float) -> float
    Función de densidad de `X`.
  g : (float) -> float
    Función de densidad de `Y`.
  c : float
    Cota superior de `f(x) / g(x)` para `x` tal que `f(x) != 0`.

  Retorna
  -------
  float
    Un valor en el rango de `X`.
  """
  x = Y_gen()
  while random() >= f(x) / (c * g(x)): x = Y_gen()
  return x