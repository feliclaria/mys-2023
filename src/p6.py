from prettytable import PrettyTable
from random import random
from scipy.integrate import quad

import continuous as cont
import math


def ex1():
  n = 0
  E_prev, E, S_sqr = 0, 0, 0

  while n < 100 or S_sqr * 10000 > n + 1:
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

  while n < 100 or S_sqr * 10000 > n + 1:
    n += 1
    theta_prev = theta
    theta += (g(1 - random()) - theta_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (theta - theta_prev)**2

  return quad(g, 0, 1)[0], theta

def ex2ii():
  g = lambda x: x**2 * math.exp(-x**2)
  h = lambda y: 2 * g((1 - y) / y) / (y**2)

  n = 0
  theta_prev, theta, S_sqr = 0, 0, 0

  while n < 100 or S_sqr * 10000 > n + 1:
    n += 1
    theta_prev = theta
    theta += (h(1 - random()) - theta_prev) / (n+1)
    S_sqr = S_sqr * (n-1) / n + (n+1) * (theta - theta_prev)**2

  return quad(h, 0, 1)[0], theta

def ex2():
  val_i, theta_i = ex2i()
  val_ii, theta_ii = ex2ii()

  table = PrettyTable(['', 'Integral (i)', 'Integral (ii)'])
  table.add_row(['Aprox. numérica', val_i, val_ii])
  table.add_row(['Monte carlo', theta_i, theta_ii])
  print(table)

def main():
  ex2()

if __name__ == '__main__':
  main()
