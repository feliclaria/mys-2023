from random import random
from discrete import randint
import math

def success_rate(sims, X, is_success, *X_params):
  """
  Probabilidad de éxito de una variable aleatoria.
  """
  return sum(is_success(X()) for _ in range(sims)) / sims

def mean(sims, X, *X_params):
  """
  Valor esperado de una variable aleatoria.
  """
  return sum(X(*X_params) for _ in range(sims)) / sims

def var(sims, X, mean):
  """
  Varianza de una variable aleatoria.
  """
  return mean(sims, lambda: (X() - mean)**2)

def std_dev(sims, X, mean):
  """
  Desviación estándar de una variable aleatoria.
  """
  return math.sqrt(var(sims, X, mean))

def monte_carlo_cont(sims, g):
  """
  Aproximación de la integral `\int_{0}^{1} fun(x) dx` usando el método de
  Monte Carlo.

  """
  return sum(g(random()) for _ in range(sims)) / sims

def monte_carlo_disc(sims, N, g):
  """
  Aproximación de la suma `\sum_{k=1}^{N} fun(k)` usando el método de
  Monte Carlo.
  """
  return sum(N * g(randint(N)) for _ in range(sims)) / sims

