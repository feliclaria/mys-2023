from random import random, expovariate
import numpy as np
from scipy.integrate import quad, dblquad

def ej2Gana():
  U = random()
  return (U < 1/2 and random() + random() >= 1) \
    or (U >= 1/2 and random() + random() + random() >= 1)

def ej2ProbGana(n):
  exitos = 0

  for _ in range(n):
    if ej2Gana():
      exitos += 1

  return exitos / n


def ej2():
  iterations = [100, 1000, 10000, 100000, 1000000]
  results = []

  for n in iterations:
    results.append(ej2ProbGana(n))

  print(f'Resultados ej. 2: {results}')


def ej3Gana():
  U = random()
  return (U < 1/3 and random() + random() <= 2) \
    or (U >= 1/3 and random() + random() + random() <= 2)

def ej3ProbGana(n):
  exitos = 0

  for _ in range(n):
    if ej3Gana():
      exitos += 1

  return exitos / n

def ej3():
  iterations = [100, 1000, 10000, 100000, 1000000]
  results = []

  for n in iterations:
    results.append(ej3ProbGana(n))

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


def monteCarlo(n, fun):
  I = 0

  for _ in range(n):
    I += fun(random())

  return I / n

def ej5a(iters):
  def g(x):
    return (1 - x**2)**(3.0/2.0)

  print("==== 5.a ====")
  print(f"Aprox. numérica: \t {quad(g, 0, 1)[0]}")
  print(f"Monte Carlo: \t\t {[monteCarlo(n, g) for n in iters]}\n")

def ej5b(iters):
  def g(x):
    return x / (x**2 + 1)

  def h(y):
    return g(y+2)

  print("==== 5.b ====")
  print(f"Aprox. numérica: \t {quad(g, 2, 3)[0]}")
  print(f"Monte Carlo: \t\t {[monteCarlo(n, h) for n in iters]}\n")

def ej5c(iters):
  def g(x):
    return x * (1+x**2)**(-2)

  def h(y):
    return g(1/y - 1) * 1/(y**2)

  print("==== 5.c ====")
  print(f"Aprox. numérica: \t {quad(g, 0, np.inf)[0]}")
  print(f"Monte Carlo: \t\t {[monteCarlo(n, h) for n in iters]}\n")

def ej5d(iters):
  def g(x):
    return np.exp(-np.square(x))

  def h(y):
    return g(1/y - 1) * 1/(y**2)

  print("==== 5.d ====")
  print(f"Aprox. numérica: \t {quad(g, -np.inf, np.inf)[0]}")
  print(f"Monte Carlo: \t\t {[2 * monteCarlo(n, h) for n in iters]}\n")

def ej5e(iters):
  def g(x, y):
    return np.exp(np.square(x+y))

  print("==== 5.e ====")
  print(f"Aprox. numérica: \t {dblquad(g, 0, 1, lambda y: 0, lambda y: 1)[0]}")
  print(f"Monte Carlo: \t\t {[monteCarlo(n, lambda V: g(random(), V)) for n in iters]}\n")

def ej5f(iters):
  def g(x, y):
    return np.exp(-(x+y))

  def h(u, z):
    return g(1/u - 1, 1/z - 1) * 1 / np.square(z*u)

  print("==== 5.f ====")
  print(f"Aprox. numérica: \t {dblquad(g, 0, np.inf, lambda y: 0, lambda y: np.inf)[0]}")
  print(f"Monte Carlo: \t\t {[monteCarlo(n, lambda V: h(random(), V)) for n in iters]}\n")

def ej5():
  iters = [100, 1000, 10000, 100000, 1000000]

  ej5a(iters)
  ej5b(iters)
  ej5c(iters)
  ej5d(iters)
  ej5e(iters)
  ej5f(iters)


def main():
  ej5()

if __name__ == '__main__':
  main()