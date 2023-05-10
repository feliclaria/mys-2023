from random import random
from prettytable import PrettyTable
import continuous as cont
import simulate as sim
import math

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
  if u < 0.0625: return math.log(2) + math.log(u) / 4
  else: return 4 * u - 0.25


def ex2_pareto(a):
  u = random()
  return 1 / ((1-u) ** (1/a))

def ex2_erlang(mu, k):
  u_prod = math.prod(random() for _ in range(k))
  return - math.log(u_prod) / mu

def ex2_weibull(lambd, beta):
  u = random()
  return lambd * (-math.log(1-u)) ** (1/beta)

def ex2():
  sims = 10_000
  a, mu, k, lambd, beta = 2, 2, 2, 1, 2

  pareto = lambda: ex2_pareto(a)
  erlang = lambda: ex2_erlang(mu, k)
  weibull = lambda: ex2_weibull(lambd, beta)

  pareto_expected_true = a / (a-1)
  erlang_expected_true = k / mu
  weibull_expected_true = lambd * math.gamma(1 + 1 / beta)

  pareto_expected_sim = sim.expected_value(sims, pareto)
  erlang_expected_sim = sim.expected_value(sims, erlang)
  weibull_expected_sim = sim.expected_value(sims, weibull)

  dists = [f'Pareto(a={a})', f'Erlang(mu={mu}, k={k})', f'Weibull(lambda={lambd}, beta={beta})']
  expected_true = [pareto_expected_true, erlang_expected_true, weibull_expected_true]
  expected_sim = [pareto_expected_sim, erlang_expected_sim, weibull_expected_sim]

  table = PrettyTable()
  table.add_column('Distribución', dists)
  table.add_column('Media exacta', expected_true)
  table.add_column(f'Media estimada en {sims} sims.', expected_sim)

  print(table)
  return


def ex3():
  return


def ex4():
  y = cont.exponential(1)
  u = random()
  return u ** (1/y)


def ej5():
  return


def ej6():
  return


def ej7():
  return


def ej8():
  return


def main():
  ex2()
  return

if __name__ == '__main__':
  main()