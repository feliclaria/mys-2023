from random import random, expovariate
from scipy.integrate import quad, dblquad
import numpy as np
import simulate as sim

iters = [100, 1_000, 10_000, 100_000, 1_000_000]


def e2_roll():
  if random() < 1/2:
    return random() + random()
  else:
    return random() + random() + random()

def e2_is_success(X):
  return X >= 1

def e2():
  results = [sim.success_prob(n, e2_roll, e2_is_success) for n in iters]
  print(f'Resultados ej. 2: {results}')


def e3_roll():
  if random() < 1/3:
    return random() + random()
  else:
    return random() + random() + random()

def e3_is_success(X):
  return X <= 2

def e3():
  results = [sim.success_prob(n, e3_roll, e3_is_success) for n in iters]
  print(f'Resultados ej. 3: {results}')


e4_probs = [0.40, 0.32, 0.28]
e4_means = [3, 4, 5]

def e4_is_box(U, box):
  a = sum(e4_probs[0:box])
  b = sum(e4_probs[0:box+1])
  return a < U <= b

def e4_box_roll():
  box = 0
  U = random()
  while not e4_is_box(U, box):
    box += 1
  return box

def e4_roll(box):
  return expovariate(1 / e4_means[box])

def e4_is_success(X):
  return X <= 4

def e4():
  n = 1_000
  total_successes = 0
  failures_per_box = [0, 0, 0]

  for _ in range(n):
    box = e4_box_roll()
    X = e4_roll(box)
    if e4_is_success(X):
      total_successes += 1
    else:
      failures_per_box[box] += 1

  print(f"Prob. tarde menos de 4 mins.: {total_successes / n}")
  for box in range(3):
    print(f"Prob. eligió caja {box} habiendo tardado más de 4 mins.: {failures_per_box[box] / (n - total_successes)}")


def e5a():
  g = lambda x: (1 - x**2)**(3.0/2.0)
  results = [sim.monte_carlo_cont(n,g) for n in iters]

  print("==== 5.a ====")
  print(f"Aprox. numérica: \t {quad(g, 0, 1)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5b():
  g = lambda x: x / (x**2 + 1)
  h = lambda y: g(y+2)
  results = [sim.monte_carlo_cont(n,h) for n in iters]

  print("==== 5.b ====")
  print(f"Aprox. numérica: \t {quad(g, 2, 3)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5c():
  g = lambda x: x * (1+x**2)**(-2)
  h = lambda y: g(1/y - 1) * 1/(y**2)
  results = [sim.monte_carlo_cont(n,h) for n in iters]

  print("==== 5.c ====")
  print(f"Aprox. numérica: \t {quad(g, 0, np.inf)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5d():
  g = lambda x: np.exp(-np.square(x))
  h = lambda y: 2 * g(1/y - 1) * 1/(y**2)
  results = [sim.monte_carlo_cont(n,h) for n in iters]

  print("==== 5.d ====")
  print(f"Aprox. numérica: \t {quad(g, -np.inf, np.inf)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5e():
  g = lambda x, y: np.exp(np.square(x+y))
  g_fixed = lambda V: g(random(), V)
  results = [sim.monte_carlo_cont(n, g_fixed) for n in iters]

  print("==== 5.e ====")
  print(f"Aprox. numérica: \t {dblquad(g, 0, 1, lambda _: 0, lambda _: 1)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5f():
  g = lambda x, y: np.exp(-(x+y))
  h = lambda u, z: g(1/u - 1, 1/z - 1) * 1 / np.square(z*u)
  h_fixed = lambda V: h(random(), V)
  results = [sim.monte_carlo_cont(n, h_fixed) for n in iters]

  print("==== 5.f ====")
  print(f"Aprox. numérica: \t {dblquad(g, 0, np.inf, lambda _: 0, lambda _: np.inf)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5():
  e5a()
  e5b()
  e5c()
  e5d()
  e5e()
  e5f()


def e6_roll():
  N = 0
  S = 0
  while S <= 1:
    S += random()
    N += 1
  return N

def e6():
  results = [sim.expected_value(n, e6_roll) for n in iters]
  print(results)


def e7_roll():
  N = 0
  S = 1
  while S >= np.exp(-3):
    S *= random()
    N += 1
  return N

def e7a():
  results = [sim.expected_value(n, e7_roll) for n in iters]
  print(results)

def e7b():
  n = 1_000_000
  results = [sim.success_prob(n, e7_roll, lambda X: X == i) for i in range(7)]
  print(results)

def e7():
  e7a()
  e7b()


def e8_roll():
  if 1/6 <= random() <= 5/6:
    return random() + random()
  else:
    return 2 * random()

def e8_is_success(X):
  return X > 5/6

def e8():
  results = [sim.success_prob(n, e8_roll, e8_is_success) for n in iters]
  print(results)


def main():
  e4()

if __name__ == '__main__':
  main()
