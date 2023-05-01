from my_random import randint
from time import time
import numpy as np
import simulate as sim

iters = [100, 1_000, 10_000, 100_000]


def ex1_roll(N=100):
  stack = list(range(N))
  cards = []
  X = 0

  for i in range(N):
    card = stack[randint(N-i) - 1]
    stack.remove(card)
    cards.append(card)
    if card == i:
      X += 1

  return X, cards

def ex1_roll_X(N=100):
  X, _ = ex1_roll(N)
  return X

def ex1_roll_cards(N=100):
  _, cards = ex1_roll(N)
  return cards

def ex1_is_success_i(cards, r=10):
  i = 0
  is_success = True

  while is_success and i < r < len(cards):
    is_success = is_success and cards[i] == i
    i += 1

  return is_success

def ex1_is_success_ii(cards, r=10):
  i = 0
  is_success = True

  while is_success and i < r < len(cards):
    is_success = is_success and cards[i] == i
    i += 1

  while is_success and i < len(cards):
    is_success = is_success and cards[i] != i
    i += 1

  return is_success

def ex1():
  results_1a_i = [sim.success_prob(n, ex1_roll_cards, ex1_is_success_i) for n in iters]
  results_1a_ii = [sim.success_prob(n, ex1_roll_cards, ex1_is_success_ii) for n in iters]
  expected_vals = [sim.expected_value(n, ex1_roll_X) for n in iters]
  expected_vals_sqrd = [sim.expected_value(n, lambda: ex1_roll_X()**2) for n in iters]
  variances = [expected_vals_sqrd[i] - expected_vals[i]**2 for i in range(len(iters))]

  print(f'Iteraciones: \t{iters}')
  print(f'Ej. 1a.i: \t{results_1a_i}')
  print(f'Ej. 1a.ii: \t{results_1a_ii}')
  print(f'Esperanza: \t{expected_vals}')
  print(f'Varianza: \t{variances}')


def ex2():
  n = 100
  N = 10_000
  f = lambda k: np.exp(k/N)

  start = time()
  monte_carlo = sim.monte_carlo_disc(n, N, f)
  print(f'Aprox. con {n} v. a.: \t{monte_carlo} \t({time() - start}s)')

  start = time()
  partial_sum = sum(f(k+1) for k in range(n))
  print(f'Primeros {n} términos: \t{partial_sum} \t({time() - start}s)')

  start = time()
  total_sum = sum(f(k+1) for k in range(N))
  print(f'Suma exacta: \t\t{total_sum} \t({time() - start}s)')


def ex3_roll():
  seen = []
  N = 0

  while len(seen) < 11:
    N += 1
    X = randint(6) + randint(6)
    if X not in seen:
      seen.append(X)

  return N

def ex3():
  expected_vals = [sim.expected_value(n, ex3_roll) for n in iters]
  expected_vals_sqrd = [sim.expected_value(n, lambda: ex3_roll()**2) for n in iters]
  variances = [expected_vals_sqrd[i] - expected_vals[i]**2 for i in range(len(iters))]
  is_success_i = [sim.success_prob(n, ex3_roll, lambda N: N >= 15) for n in iters]
  is_success_ii = [sim.success_prob(n, ex3_roll, lambda N: N <= 9) for n in iters]

  print(f'Iteraciones: \t\t{iters}')
  print(f'Esperanza: \t\t{expected_vals}')
  print(f'Desviación estándar: \t{np.sqrt(variances)}')
  print(f'Prob. >= 15: \t\t{is_success_i}')
  print(f'Prob. <= 9: \t\t{is_success_ii}')


def main():
  ex3()

if __name__ == '__main__':
  main()
