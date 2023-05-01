from random import random
from my_random import randint

def success_prob(n, roll, is_success):
  return sum(is_success(roll()) for _ in range(n)) / n

def expected_value(n, roll):
  return sum(roll() for _ in range(n)) / n

def monte_carlo_cont(n, fun):
  return sum(fun(random()) for _ in range(n)) / n

def monte_carlo_disc(n, N, fun):
  return sum(N * fun(randint(N)) for _ in range(n)) / n
