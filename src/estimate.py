from random import random
from typing import Callable

def mean(
  tol: float,
  X: Callable[..., float],
  *X_args: ...
) -> float:
  n = 1
  mean = X(*X_args)
  tol_sqr = tol**2
  mean_prev, S_sqr = 0, 0

  while n < 100 or n <= S_sqr / tol_sqr:
    n += 1
    mean_prev = mean
    mean += (X(*X_args) - mean_prev) / n
    S_sqr = S_sqr * (n-2) / (n-1) + n * (mean - mean_prev)**2

  return mean

def mean_confidence_int(
  z_alfa_2 : float,
  L : float,
  X : Callable[..., float],
  *X_args : ...
) -> float:
  return mean(L / (2 * z_alfa_2), X, *X_args)

def integral_0_to_1(
  tol: float,
  g: Callable[[float], float]
) -> float:
  return mean(tol, lambda: g (1 - random()))

def integral_a_to_b(
  tol: float,
  g: Callable[[float], float],
  a: float,
  b: float
) -> float:
  h = lambda y: g(a + (b-a) * y)
  X = lambda: h(1 - random())
  return mean(tol / (b-a), X)

def integral_0_to_inf(
  tol: float,
  g: Callable[[float], float]
) -> float:
  h = lambda y: 1 / y**2 * g((1-y) / y)
  X = lambda: h(1 - random())
  return mean(tol, X)
