from random import random
from discrete import randint
import math

def success_prob(n, random_var, is_success):
  return sum(is_success(random_var()) for _ in range(n)) / n

def expected_value(n, random_var):
  return sum(random_var() for _ in range(n)) / n

def variance(n, random_var, expected_val):
  return expected_value(n, lambda: (random_var() - expected_val)**2)

def std_deviation(n, random_var, expected_val):
  return math.sqrt(variance(n, random_var, expected_val))

def monte_carlo_cont(n, fun):
  return sum(fun(random()) for _ in range(n)) / n

def monte_carlo_disc(n, N, fun):
  return sum(N * fun(randint(N)) for _ in range(n)) / n

