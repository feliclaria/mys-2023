from random import random

def expected_value(gen_X, tol):
  n = 0
  E_prev, E, S_sqr = 0, 0, 0

  while n < 100 or S_sqr > (n+1) * tol**2:
    n += 1
    E_prev = E
    E += (gen_X() - E_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (E - E_prev)**2

  return E

def monte_carlo(h, tol):
  return expected_value(lambda: h(1 - random()), tol)