from random import random
import math


def expected_value(gen_X, tol):
  n, E_prev, E, S_sqr = 0, 0, 0, 0

  while n < 100 or n+1 < S_sqr / tol**2 :
    n += 1
    E_prev = E
    E += (gen_X() - E_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (E - E_prev)**2

  return E

def expected_value_interval(gen_X, critical_val, max_length):
  return expected_value(gen_X, max_length / (2 * critical_val))

def monte_carlo(h, tol):
  return expected_value(lambda: h(1 - random()), tol)