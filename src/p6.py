from tabulate import tabulate
from continuous import normal
from random import random
from scipy.integrate import quad
from numpy import inf
from sys import argv

import estimate
import math


def ex1():
  mean, S_sqr, n = estimate.mean(0.1, normal, 0, 1)
  table = [['size', 'mean', 'variance'], [n, mean, S_sqr]]
  print(tabulate(table, headers='firstrow'))


def ex2():
  g_i = lambda x: math.exp(x) / math.sqrt(2*x)
  g_ii = lambda x: 2 * x**2 * math.exp(-x**2)

  mean_i, _, n_i = estimate.integral_0_to_1(0.01, g_i)
  mean_ii, _, n_ii = estimate.integral_0_to_inf(0.01, g_ii)

  I_i, _ = quad(g_i, 0, 1)
  I_ii, _ = quad(g_ii, 0, inf)

  table = [
    ['integral', 'sims.', 'I', 'approx.'],
    ['(i)', n_i, mean_i, I_i],
    ['(ii)', n_ii, mean_ii, I_ii]
  ]
  print(tabulate(table, headers='firstrow', stralign='center'))


def ex3_monte_carlo(h):
  sims = [1_000, 5_000, 7_000]
  results = [['sims.', 'I', 'S', 'CI(95%)']]

  L = 0.001
  z_alpha_2 = 1.96
  tol_sqr = (L / (2*z_alpha_2))**2

  n = 1
  theta = h(1-random())
  theta_prev, S_sqr = 0, 0

  while n < 100 or n <= S_sqr / tol_sqr:
    if n in sims:
      const = 1.96 * math.sqrt(S_sqr/n)
      results.append([n, theta, math.sqrt(S_sqr), (
        round(theta - const, 4),
        round(theta + const, 4)
      )])

    n += 1
    theta_prev = theta
    theta += (h(1-random()) - theta_prev) / n
    S_sqr = S_sqr * (n-2) / (n-1) + n * (theta - theta_prev)**2

  const = 1.96 * math.sqrt(S_sqr/n)
  results.append([n, theta, math.sqrt(S_sqr), (
    round(theta - const, 4),
    round(theta + const, 4)
  )])
  return results

def ex3():
  g_i = lambda x: math.sin(x) / x
  h_i = lambda y: math.pi * g_i(math.pi * (y+1))
  results_i = ex3_monte_carlo(h_i)

  print(tabulate([[f'(3.i)']], headers='firstrow'))
  print(tabulate(results_i, headers='firstrow', stralign='center'))

  print()

  g_ii = lambda x: 3 / (3 + x**4)
  h_ii = lambda y: 1 / y**2 * g_ii((1-y) / y)
  results_ii = ex3_monte_carlo(h_ii)

  print(tabulate([[f'(3.ii)']], headers='firstrow'))
  print(tabulate(results_ii, headers='firstrow', stralign='center'))


def ex4_N():
  n, s = 0, 0
  while s <= 1:
    n += 1
    s += random()
  return n

def ex4():
  table = [['sims.', 'e', 'S', 'CI(95%)']]

  L = 0.025
  z_alpha_2 = 1.96
  tol_sqr = (L / (2*z_alpha_2))**2

  n = 1
  theta = ex4_N()
  theta_prev, S_sqr = 0, 0

  while n < 100 or n <= S_sqr / tol_sqr:
    if n == 1_000:
      const = 1.96 * math.sqrt(S_sqr/n)
      table.append([n, theta, math.sqrt(S_sqr), (
        round(theta - const, 4),
        round(theta + const, 4)
      )])

    n += 1
    theta_prev = theta
    theta += (ex4_N() - theta_prev) / n
    S_sqr = S_sqr * (n-2) / (n-1) + n * (theta - theta_prev)**2

  const = 1.96 * math.sqrt(S_sqr/n)
  table.append([n, theta, math.sqrt(S_sqr), (
    round(theta - const, 4),
    round(theta + const, 4)
  )])
  print(tabulate(table, headers='firstrow', stralign='center'))


def ex5():
  print('---')


def ex6_X():
  u = random() * 2 - 1
  v = random() * 2 - 1
  return u**2 + v**2 <= 1

def ex6():
  scale = 4

  pi_i, var_i, n_i = estimate.rate(0.01, ex6_X, scale=scale)
  interval_i = (
    round(pi_i - math.sqrt(var_i/n_i), 4),
    round(pi_i + math.sqrt(var_i/n_i), 4)
  )

  pi_ii, var_ii, n_ii = estimate.rate_interval(1.96, 0.1, ex6_X, scale=scale)
  interval_ii = (
    round(pi_ii - math.sqrt(var_ii/n_ii), 4),
    round(pi_ii + math.sqrt(var_ii/n_ii), 4)
  )

  table = [
    ['ex.', 'sims.', 'pi', 'var.', 'CI(95%)'],
    ['(a)', n_i, pi_i, var_i, interval_i],
    ['(b)', n_ii, pi_ii, var_ii, interval_ii]
  ]
  print(tabulate(table, headers='firstrow', stralign='center'))


def ex(k, fun):
  print()
  print(tabulate([[f'(Ex. {k})']]))
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