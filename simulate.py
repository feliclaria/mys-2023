from random import random

def success_prob(n, roll, is_success):
  return sum(is_success(roll()) for _ in range(n)) / n

def expected_value(n, roll):
  return sum(roll() for _ in range(n)) / n

def monte_carlo(n, fun):
  return sum(fun(random()) for _ in range(n)) / n