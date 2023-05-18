from time import time
from random import random
from prettytable import PrettyTable
import continuous as cont
import simulate as sim
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


def ex2_pareto(a):
  u = random()
  return 1 / ((1-u) ** (1/a))

def ex2_erlang(mu, k):
  u_prod = math.prod(1 - random() for _ in range(k))
  return - math.log(u_prod) / mu

def ex2_weibull(lambd, beta):
  u = random()
  return lambd * (-math.log(1-u)) ** (1/beta)

def ex2():
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


def ex3b():
  gens = [
    lambda: cont.exponential(3),
    lambda: cont.exponential(5),
    lambda: cont.exponential(7)
  ]
  probs = [0.5, 0.3, 0.2]
  return cont.composition_method(gens, probs)

def ex3b_optimized():
  u = random()
  aux = -1 * math.log(1 - random())
  if u < 0.5: return 3 * aux
  elif u < 0.8: return 5 * aux
  else: return 7 * aux

def ex3():
  exp_val = 4.4 # pre-calculated

  start = time()
  exp_val_sim = sim.expected_value(sims, ex3b)
  time_sim = time() - start

  start = time()
  exp_val_optimized = sim.expected_value(sims, ex3b_optimized)
  time_optimized = time() - start

  table = PrettyTable(['', 'E[X]', 'Tiempo (s)'])
  table.align = 'l'
  table.add_row(['Valor exacto', exp_val, '-'])
  table.add_row([f'{sims} sims.', exp_val_sim, time_sim])
  table.add_row([f'{sims} sims. (optimizado)', exp_val_optimized, time_optimized])

  print(table)
  return


def ex4():
  y = cont.exponential(1)
  u = random()
  return u ** (1/y)


def ex5_M(random_vars):
  n = len(random_vars)
  return max(random_vars[i]() for i in range(n))

def ex5_m(random_vars):
  n = len(random_vars)
  return min(random_vars[i]() for i in range(n))

def ex5():
  n = 10

  gens = [
    lambda: cont.exponential(1),
    lambda: cont.exponential(2),
    lambda: cont.exponential(3)
  ]

  M_sample = [ex5_M(gens) for _ in range(n)]
  m_sample = [ex5_m(gens) for _ in range(n)]

  table = PrettyTable()
  table.title = f'Muestra de tamaño {n}'
  table.add_column('M', M_sample)
  table.add_column('m', m_sample)
  table.add_row([f'E[M] = {sum(M_sample) / n}', f'E[m] = {sum(m_sample) / n}'])

  table_lines = table.get_string().split('\n')

  result_lines = 1
  print("\n".join(table_lines[:-(result_lines + 1)]))
  print(table_lines[2])
  print("\n".join(table_lines[-(result_lines + 1):]))

  return


def ex6_max(n):
  return max(random() for _ in range(n))

def ex6_acc_rej(n):
  x = random()
  while random() >= x**(n-1): x = random()
  return x

def ex6_inv_tr(n):
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
  sample_inv_tr = [ex6_inv_tr(n) for _ in range(sims)]
  time_inv_tr = time() - start

  table = PrettyTable(['Método', 'E[X]', 'Tiempo (s)'])
  table.align = 'l'
  table.add_row(['Máximo', sum(sample_max) / sims, time_max])
  table.add_row(['Aceptación y rechazo', sum(sample_acc_rej) / sims, time_acc_rej])
  table.add_row(['Transformada inversa', sum(sample_inv_tr) / sims, time_inv_tr])

  print(table)
  return


def ex7_inv_tr():
  return math.exp(random())

def ex7_acc_rej():
  x = random() * (math.e - 1) + 1
  while random() >= 1/x: x = random() * (math.e - 1) + 1
  return x

def ex7():
  start = time()
  sample_inv_tr = [ex7_inv_tr() for _ in range(sims)]
  time_inv_tr = time() - start
  exp_val_inv_tr = sum(sample_inv_tr) / sims
  prob_success_inv_tr = sum(x <= 2 for x in sample_inv_tr) / sims

  start = time()
  sample_acc_rej = [ex7_acc_rej() for _ in range(sims)]
  time_acc_rej = time() - start
  exp_val_acc_rej = sum(sample_acc_rej) / sims
  prob_success_acc_rej = sum(x <= 2 for x in sample_acc_rej) / sims

  table = PrettyTable(['#', 'E[X]', 'P(X<=2)', 'Tiempo (s)'])
  table.title = 'Ejercicio 7'
  table.align = 'l'
  table.add_row(['Valor exacto', math.e - 1, math.log(2), '-'])
  table.add_row(['Transformada inversa', exp_val_inv_tr, prob_success_inv_tr, time_inv_tr])
  table.add_row(['Aceptación y rechazo', exp_val_acc_rej, prob_success_acc_rej, time_acc_rej])

  print(table)
  return


def ex8_inv_tr():
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
  is_sucess = lambda X: X > 1.5

  start = time()
  sample_inv_tr = [ex8_inv_tr() for _ in range(sims)]
  time_inv_tr = time() - start
  exp_val_inv_tr = sum(sample_inv_tr) / sims
  prob_success_inv_tr = sum(is_sucess(x) for x in sample_inv_tr) / sims

  start = time()
  sample_acc_rej = [ex8_acc_rej() for _ in range(sims)]
  time_acc_rej = time() - start
  exp_val_acc_rej = sum(sample_acc_rej) / sims
  prob_success_acc_rej = sum(is_sucess(x) for x in sample_acc_rej) / sims

  table = PrettyTable(['#', 'E[X]', 'P(X>=1.5)', 'Tiempo (s)'])
  table.title = 'Ejercicio 8'
  table.align = 'l'
  table.add_row(['Valor exacto', 1, 0.125, '-'])
  table.add_row(['Transformada inversa', exp_val_inv_tr, prob_success_inv_tr, time_inv_tr])
  table.add_row(['Aceptación y rechazo', exp_val_acc_rej, prob_success_acc_rej, time_acc_rej])

  print(table)
  return


def ex9_exp(mu, sigma):
  Y1 = -math.log(1 - random())
  Y2 = -math.log(1 - random())
  while Y2 <= (Y1 - 1)**2 / 2:
    Y1 = -math.log(1 - random())
    Y2 = -math.log(1 - random())
  if random() < 0.5: return Y1 * sigma + mu
  else: return -Y1 * sigma + mu

def ex9_polar(mu, sigma):
  sqr_rad = -2 * math.log(1 - random())
  theta = random() * 2 * math.pi
  X = math.sqrt(sqr_rad) * math.cos(theta)
  Y = math.sqrt(sqr_rad) * math.sin(theta)
  return X * sigma + mu, Y * sigma + mu

def ex9_uniform(mu, sigma):
  const = 4 * math.exp(-1) / 2.0

  u = random()
  v = 1 - random()
  X = (u - 0.5) / v

  while const * X**2 > - math.log(v):
    u = random()
    v = 1 - random()
    X = (u - 0.5) / v
  return X * sigma + mu

def ex9():
  sims = 10_000
  mu = -2
  sigma = 0.5

  E = {'exp': 0, 'polar': 0, 'uniform': 0}
  Var = {'exp': 0, 'polar': 0, 'uniform': 0}
  Var_fun = lambda x, E: (x - E)**2

  for k in range(int(sims / 2)):
    E['exp'] = (E['exp'] * 2 * k + ex9_exp(mu, sigma) + ex9_exp(mu, sigma)) / (2 * k + 2)
    E['polar'] = (E['polar'] * 2 * k + sum(ex9_polar(mu, sigma))) / (2 * k + 2)
    E['uniform'] = (E['uniform'] * 2 * k + ex9_uniform(mu, sigma) + ex9_uniform(mu, sigma)) / (2 * k + 2)

    Var['exp'] = (Var['exp'] * 2 * k + Var_fun(ex9_exp(mu, sigma), E['exp']) + Var_fun(ex9_exp(mu, sigma), E['exp'])) / (2 * k + 2)
    Var['polar'] = (Var['polar'] * 2 * k + sum(map(lambda x: Var_fun(x, E['polar']), ex9_polar(mu, sigma)))) / (2 * k + 2)
    Var['uniform'] = (Var['uniform'] * 2 * k + Var_fun(ex9_exp(mu, sigma), E['uniform']) + Var_fun(ex9_exp(mu, sigma), E['exp'])) / (2 * k + 2)

  table = PrettyTable(['#', 'E[X]', 'Var[X]'])
  table.title = 'Ejercicio 9'
  table.align = 'l'
  table.add_row(['Valor exacto', mu, sigma**2])
  table.add_row(['Gen. de exponenciales', E['exp'], Var['exp']])
  table.add_row(['Método polar', E['polar'], Var['polar']])
  table.add_row(['Razón de uniformes', E['uniform'], Var['uniform']])
  print(table)


def main():
  ex9()
  return

if __name__ == '__main__':
  main()