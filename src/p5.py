from discrete import randint
from random import random
from simulate import mean
from tabulate import tabulate
from time import time
from sys import argv

import poisson
import continuous as cont
import numpy as np
import math

sims = 10_000

def ex1a():
  u = random()
  if u < 0.25: return 2 + 2 * math.sqrt(u)
  else: return 6 - 6 * math.sqrt((1-u) / 3)

def ex1b():
  u = random()
  if u < 0.6: return math.sqrt(324 + 420 * u) / 6 - 3
  else: return math.cbrt((35 * u - 19) / 2)

def ex1c():
  u = random()
  if u < 0.0625: return math.log(2) + math.log(1 - u) / 4
  else: return 4 * u - 0.25


def pareto(a):
  u = random()
  return 1 / ((1-u) ** (1/a))

def erlang(mu, k):
  u_prod = math.prod(1 - random() for _ in range(k))
  return - math.log(u_prod) / mu

def weibull(lambd, beta):
  u = random()
  return lambd * (-math.log(1-u)) ** (1/beta)

def ex2():
  a, mu, k, lambd, beta = 2, 2, 2, 1, 2

  table = [
    ['dist', 'real mean', f'{sims} sims.'],
    [f'Pareto({a})', a / (a-1), mean(sims, pareto, a)],
    [f'Erlang{mu, k}', k / mu, mean(sims, erlang, mu, k)],
    [
      f'Weibull{lambd, beta}',
      lambd * math.gamma(1 + 1 / beta),
      mean(sims, weibull, lambd, beta)
    ]
  ]
  print(tabulate(table, headers='firstrow'))


def ex3_X():
  gens = [
    lambda: cont.exponential(1/3),
    lambda: cont.exponential(1/5),
    lambda: cont.exponential(1/7)
  ]
  probs = [0.5, 0.3, 0.2]
  return cont.composition_method(gens, probs)

def ex3_X_optimized():
  u = random()
  aux = -1 * math.log(1 - random())
  if u < 0.5: return 3 * aux
  elif u < 0.8: return 5 * aux
  else: return 7 * aux

def ex3():
  start = time()
  mean_sim = mean(sims, ex3_X)
  end = time()
  time_sim = end - start


  start = time()
  mean_optimized = mean(sims, ex3_X_optimized)
  end = time()
  time_optimized = end - start

  table = [
    ['method', 'mean', 'time (s)'],
    ['exact', 4.4],
    ['composition', mean_sim, time_sim],
    ['composition (optimized)', mean_optimized, time_optimized]
  ]
  print(tabulate(table, headers='firstrow'))


def ex4_X():
  y = cont.exponential(1)
  u = random()
  return u ** (1/y)

def ex4():
  start = time()
  mean_sim = mean(sims, ex4_X)
  time_sim = time() - start

  table = [
    [f'{sims} sims.', 'time (s)'],
    [mean_sim, time_sim],
  ]
  print(tabulate(table, headers='firstrow'))


def ex5_M(gens):
  return max(X() for X in gens)

def ex5_m(gens):
  return min(X() for X in gens)

def ex5():
  size = 10
  gens = [
    lambda: cont.exponential(1),
    lambda: cont.exponential(2),
    lambda: cont.exponential(3)
  ]

  M_sample = [ex5_M(gens) for _ in range(size)]
  m_sample = [ex5_m(gens) for _ in range(size)]

  data_table = [['M', 'm']]
  for M_i, m_i in zip(M_sample, m_sample):
    data_table.append([M_i, m_i])

  mean_table = [
    ['sample', 'mean'],
    ['M', sum(M_sample) / size],
    ['m', sum(m_sample) / size]
  ]

  print(tabulate(data_table, headers='firstrow'))
  print()
  print(tabulate(mean_table, headers='firstrow'))


def ex6_max(n):
  return max(random() for _ in range(n))

def ex6_acc_rej(n):
  x = random()
  while random() >= x**(n-1): x = random()
  return x

def ex6_inv_trans(n):
  return random() ** (1/n)

def ex6():
  n = 10

  start = time()
  sample_max = [ex6_max(n) for _ in range(sims)]
  time_max = time() - start

  start = time()
  sample_acc_rej = [ex6_acc_rej(n) for _ in range(sims)]
  time_acc_rej = time() - start

  start = time()
  sample_inv_trans = [ex6_inv_trans(n) for _ in range(sims)]
  time_inv_trans = time() - start

  table = [
    ['method', 'mean', 'time (s)'],
    ['maximum', sum(sample_max) / sims, time_max],
    ['accept-reject', sum(sample_acc_rej) / sims, time_acc_rej],
    ['inverse transform', sum(sample_inv_trans) / sims, time_inv_trans],
  ]
  print(tabulate(table, headers='firstrow'))


def ex7_inv_trans():
  return math.exp(random())

def ex7_acc_rej():
  x = random() * (math.e - 1) + 1
  while random() >= 1/x: x = random() * (math.e - 1) + 1
  return x

def ex7():
  start = time()
  sample_inv_trans = [ex7_inv_trans() for _ in range(sims)]
  end = time()

  inv_trans = [
    'inverse transform',
    sum(sample_inv_trans) / sims,
    sum(x <= 2 for x in sample_inv_trans) / sims,
    end - start
  ]

  start = time()
  sample_acc_rej = [ex7_acc_rej() for _ in range(sims)]
  end = time()

  acc_rej = [
    'accept-reject',
    sum(sample_acc_rej) / sims,
    sum(x <= 2 for x in sample_acc_rej) / sims,
    end - start
  ]

  table = [
    ['method', 'mean', 'P(X<=2)', 'time (s)'],
    ['exact', math.e - 1, math.log(2)],
    inv_trans,
    acc_rej
  ]
  print(tabulate(table, headers='firstrow'))


def ex8_inv_trans():
  u = random()
  if u < 0.5: return math.sqrt(2 * u)
  else: return 2 - math.sqrt(2 - 2 * u)

def ex8_acc_rej():
  x = random() * 2
  u = random()
  while (x < 1 and u >= x) or u >= 2 - x:
    x = random() * 2
    u = random()
  return x

def ex8():
  start = time()
  sample_inv_trans = [ex8_inv_trans() for _ in range(sims)]
  end = time()

  inv_trans = [
    'inverse transform',
    sum(sample_inv_trans) / sims,
    sum(x >= 1.5 for x in sample_inv_trans) / sims,
    end - start
  ]

  start = time()
  sample_acc_rej = [ex8_acc_rej() for _ in range(sims)]
  end = time()

  acc_rej = [
    'accept-reject',
    sum(sample_acc_rej) / sims,
    sum(x >= 1.5 for x in sample_acc_rej) / sims,
    end - start
  ]

  table = [
    ['method', 'mean', 'P(X>=1.5)', 'time (s)'],
    ['exact', 1, 0.125],
    inv_trans,
    acc_rej
  ]
  print(tabulate(table, headers='firstrow'))


def normal_exp(mu, sigma):
  y1 = -math.log(1 - random())
  y2 = -math.log(1 - random())
  while y2 <= (y1 - 1)**2 / 2:
    y1 = -math.log(1 - random())
    y2 = -math.log(1 - random())
  if random() < 0.5: return y1 * sigma + mu
  else: return -y1 * sigma + mu

def normal_polar(mu, sigma):
  r_sqr = -2 * math.log(1 - random())
  theta = random() * 2 * math.pi
  x = math.sqrt(r_sqr) * math.cos(theta)
  y = math.sqrt(r_sqr) * math.sin(theta)
  return x * sigma + mu, y * sigma + mu

def normal_ratio(mu, sigma):
  return cont.normal(mu, sigma)

def ex9():
  mu, sigma = -2, 0.5

  start = time()
  sample_exp = [normal_exp(mu, sigma) for _ in range(sims)]
  end = time()
  exponentials = [
    'gen. of exponentials',
    np.mean(sample_exp),
    np.var(sample_exp, ddof=1),
    end - start
  ]

  start = time()
  sample_polar = [x for _ in range(sims // 2) for x in normal_polar(mu, sigma)]
  end = time()
  polar_coords = [
    'polar coordinates',
    np.mean(sample_polar),
    np.var(sample_polar, ddof=1),
    end - start
  ]

  start = time()
  sample_ratio = [normal_ratio(mu, sigma) for _ in range(sims)]
  end = time()
  ratio_uniforms = [
    'ratio of uniforms',
    np.mean(sample_ratio),
    np.var(sample_ratio, ddof=1),
    end - start
  ]

  table = [
    ['method', 'mean', 'var.', 'time (s)'],
    ['exact', mu, sigma**2],
    exponentials,
    polar_coords,
    ratio_uniforms
  ]
  print(tabulate(table, headers='firstrow'))


def ex14():
  lambd, t = 5, 1

  buses, times = poisson.homogeneous(lambd, t)
  capacities = [randint(40, 20) for _ in range(buses)]

  data_table = [['bus', 'capacity', 'time of arrival']]
  for capacity, time in zip(capacities, times):
    data_table.append([capacity, time])
  print(tabulate(data_table, headers='firstrow', showindex=range(1, buses+1)))

  print(f'\ntotal passengers: {sum(capacities)}')


def ex15a(T):
  lambd_t = lambda t: 3 + 4 / (t+1) if 0 <= t <= 3 else 0
  lambd = 7 # at t = 0
  return poisson.inhomogeneous(lambd, lambd_t, T)

def ex15a_improved(T):
  interv = [1, 2, 3]
  lambd = [7, 5, 13/3]
  return poisson.inhomogeneous_improved(lambd, interv, T)

def ex15b(T):
  lambd_t = lambda t: (t - 2)**2 - 5 * t + 17 if 0 <= t <= 5 else 0
  lambd = 21 # at t = 0
  return poisson.inhomogeneous(lambd, lambd_t, T)

def ex15b_improved(T):
  interv = [2, 4, 5]
  lambd = [21, 7, 1]
  return poisson.inhomogeneous_improved(lambd, interv, T)

def ex15c(T):
  def lambd_t(t):
    if 2 <= t <= 3: return t/2 - 1
    if 3 <= t <= 6: return 1 - t/6
    return 0
  lambd = 1/2 # at t = 3
  return poisson.inhomogeneous(lambd, lambd_t, T)

def ex15c_improved(T):
  interv = [3, 4, 5, 6]
  lambd = [1/2, 1/2, 1/3, 2/6]
  return poisson.inhomogeneous_improved(lambd, interv, T)

def exNone():
  pass

def ex(k, fun):
  print()
  print(tabulate([[f'(Ex. {k})']]))
  print()
  fun()
  print()

def main(argv):
  k = int(argv[1])
  funs = {
    '2': ex2,
    '3': ex3,
    '4': ex4,
    '5': ex5,
    '6': ex6,
    '7': ex7,
    '8': ex8,
    '9': ex9,
    '14': ex14,
  }

  if k == 0:
    for key, fun in funs.items(): ex(int(key), fun)
  else: ex(k, funs.get(str(k), exNone))

if __name__ == '__main__':
  main(argv)