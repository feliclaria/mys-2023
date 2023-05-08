from random import random
from discrete import randint
import math

def success_prob(sims, random_var, is_success):
  """
  Probabilidad de éxito de una variable aleatoria.
  """
  return sum(is_success(random_var()) for _ in range(sims)) / sims

def expected_value(sims, random_var):
  """
  Valor esperado de una variable aleatoria.
  """
  return sum(random_var() for _ in range(sims)) / sims

def variance(sims, random_var, expected_val):
  """
  Varianza de una variable aleatoria.
  """
  return expected_value(sims, lambda: (random_var() - expected_val)**2)

def std_deviation(sims, random_var, expected_val):
  """
  Desviación estándar de una variable aleatoria.
  """
  return math.sqrt(variance(sims, random_var, expected_val))

def monte_carlo_cont(sims, fun):
  """
  Aproximación de la integral `\int_{0}^{1} fun(x) dx` usando el método de
  Monte Carlo.

  """
  return sum(fun(random()) for _ in range(sims)) / sims

def monte_carlo_disc(sims, N, fun):
  """
  Aproximación de la suma `\sum_{k=1}^{N} fun(k)` usando el método de
  Monte Carlo.
  """
  return sum(N * fun(randint(N)) for _ in range(sims)) / sims

