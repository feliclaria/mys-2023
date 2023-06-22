from random import random
from typing import Callable, Tuple

def mean(
  tol: float,
  X: Callable[..., float],
  *X_args: ...
) -> Tuple[float, float, int]:
  n = 1
  mean = X(*X_args)
  mean_prev, S_sqr = 0, 0
  tol_sqr = tol**2

  while n < 100 or n <= S_sqr / tol_sqr:
    n += 1
    mean_prev = mean
    mean += (X(*X_args) - mean_prev) / n
    S_sqr = S_sqr * (n-2) / (n-1) + n * (mean - mean_prev)**2

  return mean, S_sqr, n

def mean_interval(
  z_alpha_2 : float,
  L : float,
  X : Callable[..., float],
  *X_args : ...
) -> Tuple[float, float, int]:
  return mean(L / (2 * z_alpha_2), X, *X_args)

def rate(
  tol: float,
  X: Callable[..., float],
  *X_args: ...,
  scale: float = 1
) -> Tuple[float, int]:
  n = 1
  p = X(*X_args)

  tol_sqr = tol**2
  scale_sqr = scale**2

  while n < 100 or n <= scale_sqr * p * (1 - p) / tol_sqr :
    n += 1
    p += (X(*X_args) - p) / n

  return scale * p, scale_sqr * p * (1-p), n

def rate_interval(
  z_alpha_2 : float,
  L : float,
  X : Callable[..., float],
  *X_args : ...,
  scale: float = 1
) -> Tuple[float, int]:
  return rate(L / (2 * z_alpha_2), X, *X_args, scale=scale)

def integral_0_to_1(
  tol: float,
  g: Callable[[float], float]
) -> Tuple[float, float, int]:
  return mean(tol, lambda: g (1 - random()))

def integral_a_to_b(
  tol: float,
  g: Callable[[float], float],
  a: float,
  b: float
) -> Tuple[float, float, int]:
  h = lambda y: g(a + (b-a) * y)
  X = lambda: h(1 - random())
  return mean(tol / (b-a), X)

def integral_0_to_inf(
  tol: float,
  g: Callable[[float], float]
) -> Tuple[float, float, int]:
  h = lambda y: 1 / y**2 * g((1-y) / y)
  X = lambda: h(1 - random())
  return mean(tol, X)
