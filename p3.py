from random import random, expovariate
import numpy as np
from scipy.integrate import quad, dblquad

iters = [100, 1000, 10000, 100000, 1000000]

def sim_success_prob(n, roll, is_success):
  return sum(is_success(roll()) for _ in range(n)) / n


def e2_roll():
  if random() < 1/2:
    return random() + random()
  else:
    return random() + random() + random()

def e2_is_success(X):
  return X >= 1

def e2():
  results = [sim_success_prob(n, e2_roll, e2_is_success) for n in iters]
  print(f'Resultados ej. 2: {results}')


def e3_roll():
  if random() < 1/3:
    return random() + random()
  else:
    return random() + random() + random()

def e3_is_success(X):
  return X <= 2

def e3():
  results = [sim_success_prob(n, e3_roll, e3_is_success) for n in iters]
  print(f'Resultados ej. 3: {results}')


def esCaja1(U):
  return U < 0.40

def esCaja2(U):
  return 0.40 <= U < 0.72

def esCaja3(U):
  return 0.72 < U

def tardaPocoDadaCaja(lambd):
  return expovariate(lambd) <= 4

def ej4():
  n = 1000
  tardanPoco = 0
  cantidadesCajas = [0, 0, 0]

  for _ in range(n):
    U = random()
    if esCaja1(U):
      if tardaPocoDadaCaja(1/3):
        tardanPoco += 1
      else:
        cantidadesCajas[0] += 1
    elif esCaja2(U):
      if tardaPocoDadaCaja(1/4):
        tardanPoco += 1
      else:
        cantidadesCajas[1] += 1
    elif esCaja3(U):
      if tardaPocoDadaCaja(1/5):
        tardanPoco += 1
      else:
        cantidadesCajas[2] += 1

  print(f"Prob. tarde menos de 4 mins.: {tardanPoco / n}")
  print(f"Prob. eligió caja 1 al tardar más de 4 mins.: {cantidadesCajas[0] / (n - tardanPoco)}")
  print(f"Prob. eligió caja 2 al tardar más de 4 mins.: {cantidadesCajas[1] / (n - tardanPoco)}")
  print(f"Prob. eligió caja 3 al tardar más de 4 mins.: {cantidadesCajas[2] / (n - tardanPoco)}")


def monte_carlo(n, fun):
  I = 0
  for _ in range(n):
    I += fun(random())
  return I / n


def e5a():
  g = lambda x: (1 - x**2)**(3.0/2.0)
  results = [monte_carlo(n,g) for n in iters]

  print("==== 5.a ====")
  print(f"Aprox. numérica: \t {quad(g, 0, 1)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5b():
  g = lambda x: x / (x**2 + 1)
  h = lambda y: g(y+2)
  results = [monte_carlo(n,h) for n in iters]

  print("==== 5.b ====")
  print(f"Aprox. numérica: \t {quad(g, 2, 3)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5c():
  g = lambda x: x * (1+x**2)**(-2)
  h = lambda y: g(1/y - 1) * 1/(y**2)
  results = [monte_carlo(n,h) for n in iters]

  print("==== 5.c ====")
  print(f"Aprox. numérica: \t {quad(g, 0, np.inf)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5d():
  g = lambda x: np.exp(-np.square(x))
  h = lambda y: 2 * g(1/y - 1) * 1/(y**2)
  results = [monte_carlo(n,h) for n in iters]

  print("==== 5.d ====")
  print(f"Aprox. numérica: \t {quad(g, -np.inf, np.inf)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5e():
  g = lambda x, y: np.exp(np.square(x+y))
  g_fixed = lambda V: g(random(), V)
  results = [monte_carlo(n, g_fixed) for n in iters]

  print("==== 5.e ====")
  print(f"Aprox. numérica: \t {dblquad(g, 0, 1, lambda _: 0, lambda _: 1)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def e5f():
  g = lambda x, y: np.exp(-(x+y))
  h = lambda u, z: g(1/u - 1, 1/z - 1) * 1 / np.square(z*u)
  h_fixed = lambda V: h(random(), V)
  results = [monte_carlo(n, h_fixed) for n in iters]

  print("==== 5.f ====")
  print(f"Aprox. numérica: \t {dblquad(g, 0, np.inf, lambda _: 0, lambda _: np.inf)[0]}")
  print(f"Monte Carlo: \t\t {results}\n")

def ej5():
  e5a()
  e5b()
  e5c()
  e5d()
  e5e()
  e5f()


def sim_expected_value(n, roll):
  return sum(roll() for _ in range(n)) / n

def e6_roll():
  N = 0
  S = 0
  while S <= 1:
    S += random()
    N += 1
  return N

def ej6():
  results = [sim_expected_value(n, e6_roll) for n in iters]
  print(results)


def e7_roll():
  N = 0
  S = 1
  while S >= np.exp(-3):
    S *= random()
    N += 1
  return N

def e7a():
  results = [sim_expected_value(n, e7_roll) for n in iters]
  print(results)

def e7b():
  n = 1000000
  results = [sim_success_prob(n, e7_roll, lambda X: X == i) for i in range(7)]
  print(results)

def ej7():
  e7a()
  e7b()


def e8_roll():
  if 1/6 <= random() <= 5/6:
    return random() + random()
  else:
    return 2 * random()

def e8_is_success(X):
  return X > 5/6

def ej8():
  results = [sim_success_prob(n, e8_roll, e8_is_success) for n in iters]
  print(results)


def main():
  e2()

if __name__ == '__main__':
  main()