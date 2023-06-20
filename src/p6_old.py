from prettytable import PrettyTable
from continuous import normal
from random import random
from scipy.integrate import quad
from numpy import inf
from sys import argv

import math


def ex1():
  n = 1
  mean = normal(0, 1)
  mean_prev, S_sqr = 0, 0

  while n < 100 or n <= S_sqr * 100:
    n += 1
    mean_prev = mean
    mean += (normal(0, 1) - mean_prev) / n
    S_sqr = S_sqr * (n-2) / (n-1) + n * (mean - mean_prev)**2

  table = PrettyTable(['Nro. sims.', 'Media muestral', 'Varianza muestral'])
  table.title = 'Ejercicio 1'
  table.add_row([n, mean, S_sqr])
  print(table)


def ex2_monte_carlo(h):
  n = 1
  theta = h(1-random())
  theta_prev, S_sqr = 0, 0

  while n < 100 or n <= S_sqr * 10_000:
    n += 1
    theta_prev = theta
    theta += (h(1-random()) - theta_prev) / n
    S_sqr = S_sqr * (n-2) / (n-1) + n * (theta - theta_prev)**2

  return n, theta

def ex2i():
  g = lambda x: math.exp(x) / math.sqrt(2 * x)
  n, theta = ex2_monte_carlo(g)
  value, _ = quad(g, 0, 1)
  return n, theta, value

def ex2ii():
  g = lambda x: x**2 * math.exp(-x**2)
  h = lambda y: 2 * g((1-y) / y) / (y**2)
  n, theta = ex2_monte_carlo(h)
  value, _ = quad(g, -inf, inf)
  return n, theta, value

def ex2():
  table = PrettyTable(['Integral', 'Nro. sims.', 'Media muestral', 'Valor aprox.'])
  table.title = 'Ejercicio 2'
  table.add_row(['(i)', *ex2i()])
  table.add_row(['(ii)', *ex2ii()])
  print(table)


def ex3_monte_carlo(h):
  sims = [1_000, 5_000, 7_000]
  results = []

  n = 1
  theta = h(1-random())
  theta_prev, S_sqr = 0, 0

  while n < 100 or n <= 3_841_600 * S_sqr:
    if n in sims:
      const = 1.96 * math.sqrt(S_sqr/n)
      results.append((n, round(theta, 4), round(math.sqrt(S_sqr), 4),
                      (round(theta - const, 4), round(theta + const, 4))
                      ))
    n += 1
    theta_prev = theta
    theta += (h(1-random()) - theta_prev) / n
    S_sqr = S_sqr * (n-2) / (n-1) + n * (theta - theta_prev)**2

  const = 1.96 * math.sqrt(S_sqr/n)
  results.append((n, round(theta, 4), round(math.sqrt(S_sqr), 4),
                  (round(theta - const, 4), round(theta + const, 4))
                  ))
  return results

def ex3i():
  g = lambda x: math.sin(x) / x
  h = lambda y: math.pi * g(math.pi * (y+1))

  table = PrettyTable(['Nro. sims.', 'I', 'S', 'IC(95%)'])
  table.title = 'Ejercicio 3i'
  table.add_rows(ex3_monte_carlo(h))
  print(table)

def ex3ii():
  g = lambda x: 3 / (3 + x**4)
  h = lambda y: 1 / y**2 * g((1-y) / y)

  table = PrettyTable(['Nro. sims.', 'I', 'S', 'IC(95%)'])
  table.title = 'Ejercicio 3ii'
  table.add_rows(ex3_monte_carlo(h))
  print(table)

def ex3():
  ex3i()
  ex3ii()


def ex4_N():
  N, S = 0, 0
  while S <= 1:
    N += 1
    S += random()
  return N

def ex4():
  n = 1
  e = ex4_N()
  e_prev, S_sqr = 0, 0

  results = []

  while n < 100 or n <= 24_586.24 * S_sqr:
    if n == 1_000:
      const = 1.96 * math.sqrt(S_sqr/n)
      results.append((n, round(e, 4), round(math.sqrt(S_sqr/n), 4),
                      (round(e - const, 4), round(e + const, 4))
                      ))
    n += 1
    e_prev = e
    e += (ex4_N() - e_prev) / n
    S_sqr = S_sqr * (n-2) / (n-1) + n * (e - e_prev)**2

  const = 1.96 * math.sqrt(S_sqr/n)
  results.append((n, round(e, 4), round(math.sqrt(S_sqr/n), 4),
                  (round(e - const, 4), round(e + const, 4))
                  ))

  table = PrettyTable(['Nro. sims', 'e', 'S', 'IC(95%)'])
  table.title = 'Ejercicio 4'
  table.add_rows(results)
  print(table)


def ex5():
  pass


def ex6_X():
  u = random() * 2 - 1
  v = random() * 2 - 1
  return u**2 + v**2 <= 1

def ex6a():
  n, p_prev, p = 0, 0, 0
  while n < 100 or n <= p * (1-p) * 160_000:
    n += 1
    p_prev = p
    p += (ex6_X() - p_prev) / n

  pi =  4 * p
  s = 4 * math.sqrt(p * (1-p) / n)
  const = 7.84 * s
  return n, round(pi, 4), round(s, 4), (round(pi - const, 4), round(pi + const, 4))

def ex6b():
  n, p, p_prev = 0, 0, 0
  while n < 100 or n <= 11333.12 * p * (1-p):
    n += 1
    p_prev = p
    p += (ex6_X() - p_prev) / n

  pi =  4 * p
  s = 4 * math.sqrt(p * (1-p) / n)
  const = 7.84 * s
  return n, round(pi, 4), round(s, 4), (round(pi - const, 4), round(pi + const, 4))

def ex6():
  table = PrettyTable(['Inciso', 'Nro. sims.', 'pi', 'S', 'IC(95%)'])
  table.title = 'Ejercicio 6'
  table.add_rows([['(a)', *ex6a()], ['(b)', *ex6b()]])
  print(table)


def ex(k, fun):
  print()
  print(f'+--- (Ex. {k}) ---+')
  print()
  fun()
  print()

def main(argv):
  k = int(argv[1])
  funs = [ex1, ex2, ex3, ex4, ex5, ex6]

  if k == 0:
    for i, fun in enumerate(funs): ex(i+1, fun)
  else: ex(k, funs[k-1])

if __name__ == '__main__':
  main(argv)