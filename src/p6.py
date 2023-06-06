from prettytable import PrettyTable
from random import random
from scipy.integrate import quad

import continuous as cont
import math


def ex1():
  n = 0
  E_prev, E, S_sqr = 0, 0, 0

  while n < 100 or n < S_sqr * 100:
    n += 1
    E_prev = E
    E += (cont.normal(0, 1) - E_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (E - E_prev)**2

  table = PrettyTable(['Tamaño de la muestra', 'Media muestral', 'Varianza muestral'])
  table.add_row([n, E, S_sqr])
  print(table)


def ex2i():
  g = lambda x: math.exp(x) / math.sqrt(2 * x)

  n = 0
  theta_prev, theta, S_sqr = 0, 0, 0

  while n < 100 or n < S_sqr * 10_000:
    n += 1
    theta_prev = theta
    theta += (g(1-random()) - theta_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (theta - theta_prev)**2

  return quad(g, 0, 1)[0], theta, n

def ex2ii():
  g = lambda x: x**2 * math.exp(-x**2)
  h = lambda y: 2 * g((1-y) / y) / (y**2)

  n = 0
  theta_prev, theta, S_sqr = 0, 0, 0

  while n < 100 or n < S_sqr * 10_000:
    n += 1
    theta_prev = theta
    theta += (h(1-random()) - theta_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (theta - theta_prev)**2

  return quad(h, 0, 1)[0], theta, n

def ex2():
  val_i, theta_i, n_i = ex2i()
  val_ii, theta_ii, n_ii = ex2ii()

  table = PrettyTable(['', 'Integral (i)', 'Integral (ii)'])
  table.add_row(['Aprox. numérica', val_i, val_ii])
  table.add_row(['Monte carlo', theta_i, theta_ii])
  table.add_row(['Nro. sims.', n_i, n_ii])
  print(table)


def ex3i():
  g = lambda x: math.sin(x) / x
  h = lambda y: math.pi * g(math.pi * (y+1))

  sims = [1_000, 5_000, 7_000]
  I_list = [sum(h(1-random()) for _ in range(n)) / n for n in sims]
  S_sqr_list = [sum((h(1-random()) - I)**2 for _ in range(n)) / n for n, I in zip(sims, I_list)]
  IC_list = [(I - 1.96 * math.sqrt(S_sqr/n), I + math.sqrt(S_sqr/n)) for n, I, S_sqr in zip(sims, I_list, S_sqr_list)]

  results = list(zip(sims, I_list, S_sqr_list, IC_list))

  n, theta_prev, theta, S_sqr = 0, 0, 0, 0
  while n < 100 or n < 3_841_600 * S_sqr:
    n += 1
    theta_prev = theta
    theta += (h(1-random()) - theta_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (theta - theta_prev)**2
  IC = theta - 1.96 * math.sqrt(S_sqr/n), theta + 1.96 * math.sqrt(S_sqr/n)

  results.append((n, theta, S_sqr, IC))

  table = PrettyTable(['Nro. sims.', 'I', 'S', 'IC(95%)'])
  table.title = 'Ejercicio 3i'
  for n, I, S_sqr, IC in results:
    table.add_row([n, round(I, 4), round(math.sqrt(S_sqr), 4), tuple(map(lambda x: round(x, 4), IC))])

  print(table)

def ex3ii():
  g = lambda x: 3 / (3 + x**4)
  h = lambda y: 1 / y**2 * g((1-y) / y)

  sims = [1_000, 5_000, 7_000]
  I_list = [sum(h(1-random()) for _ in range(n)) / n for n in sims]
  S_sqr_list = [sum((h(1-random()) - I)**2 for _ in range(n)) / n for n, I in zip(sims, I_list)]
  IC_list = [(I - 1.96 * math.sqrt(S_sqr/n), I + math.sqrt(S_sqr/n)) for n, I, S_sqr in zip(sims, I_list, S_sqr_list)]

  results = list(zip(sims, I_list, S_sqr_list, IC_list))

  n, theta_prev, theta, S_sqr = 0, 0, 0, 0
  while n < 100 or n < 3_841_600 * S_sqr:
    n += 1
    theta_prev = theta
    theta += (h(1-random()) - theta_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (theta - theta_prev)**2
  IC = theta - 1.96 * math.sqrt(S_sqr/n), theta + 1.96 * math.sqrt(S_sqr/n)

  results.append((n, theta, S_sqr, IC))

  table = PrettyTable(['Nro. sims.', 'I', 'S', 'IC(95%)'])
  table.title = 'Ejercicio 3ii'
  for n, I, S_sqr, IC in results:
    table.add_row([n, round(I, 4), round(math.sqrt(S_sqr), 4), tuple(map(lambda x: round(x, 4), IC))])

  print(table)

def ex3():
  ex3i()
  ex3ii()


def ex4_N():
  N = 0
  S = 0
  while S <= 1:
    N += 1
    S += random()
  return N

def ex4():
  n, e_prev, e, S_sqr = 0, 0, 0, 0
  while n < 100 or n < 24_586.24 * S_sqr:
    n += 1
    e_prev = e
    e += (ex4_N() - e_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (e - e_prev)**2

  table = PrettyTable(['n', 'ê', 'S²(n)'])
  table.title = 'Ejercicio 4'
  table.add_row([n, e, S_sqr])
  print(table)


def main():
  ex1()

if __name__ == '__main__':
  main()
