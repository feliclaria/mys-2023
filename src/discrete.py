from random import random
from functools import reduce
from operator import iconcat
import math

def randint(N):
  """
  Una variable aleatoria discreta con distribución uniforme.
  """
  return int(N * random()) + 1

def binomial(n, p):
  """
  Una variable aleatoria discreta con distribución binomial.

  Generada usando el método de la transformada inversa con la definición
  recusiva de su función de probabilidad de masa.
  """
  i = 0
  prob_i = (1 - p)**n
  F = prob_i
  U = random()

  while U >= F:
    i += 1
    prob_i *= (n-i-1) * p / (i * (1-p))
    F += prob_i

  return i

def negative_binomial(r, p):
  """
  Una variable aleatoria discreta con distribución binomial negativa.

  Generada usando el método de la transformada inversa con la definición
  recusiva de su función de probabilidad de masa.
  """
  i = r
  prob_i = p**r
  F = p
  U = random()

  while U >= F:
    i += 1
    prob_i *= (r+i-1) * (1-p) / i
    F += p

  return i

def poisson(lambd):
  """
  Una variable aleatoria discreta con distribución de Poisson.

  Generada usando el método de la transformada inversa con la definición
  recusiva de su función de probabilidad de masa.
  """
  i = 0
  prob_i = math.exp(-lambd)
  F = prob_i
  U = random()

  while U >= F:
    i += 1
    prob_i *= lambd / i
    F += prob_i

  return i

def poisson_fast(lambd):
  """
  Una variable aleatoria discreta con distribución de Poisson.

  Generada usando el método de la transformada inversa con la definición
  recusiva de su función de probabilidad de masa, y búsqueda lineal
  ascendente/descerndente.
  """
  prob_i = math.exp(-lambd)
  F = prob_i

  for i in range(1, int(lambd)+1):
    prob_i *= lambd / i
    F += prob_i

  U = random()

  if U >= F:
    i = int(lambd) + 1

    while U >= F:
      prob_i *= lambd / i
      F += prob_i
      i += 1

    return i - 1
  else:
    i = int(lambd)

    while F > U:
      F -= prob_i
      prob_i *= i / lambd
      i -= 1

    return i + 1

def geometric(p):
  """
  Una variable aleatoria discreta con distribución geométrica.

  Generada usando el método de la transformada inversa con la definición
  recusiva de su función de probabilidad de masa, y búsqueda lineal
  ascendente/descerndente.
  """
  i = 1
  prob_i = p
  F = prob_i
  U = random()

  while U >= F:
    i += 1
    prob_i *= (1-p)
    F += prob_i

  return i

def hypergeometric(n, N, M):
  """
  Una variable aleatoria discreta con distribución hipergeométrica.

  Generada usando el método de la transformada inversa con la definición
  recusiva de su función de probabilidad de masa, y búsqueda lineal
  ascendente/descerndente.
  """
  i = 0
  prob_i = math.comb(M-N, n) / math.comb(M, n)
  F = prob_i
  U = random()

  while U >= F:
    i += 1
    prob_i *= (N-(i-1)) * (n-(i-1)) / (i * (i + M - N - n))
    F += prob_i

  return i

def inverse_trans_fun(pmf, values):
  """
  Produce una función que genera una variable aleatoria discreta de valores
  finitos.

  Generada usando el método de la transformada inversa dado que se conocen
  los valores de su imagen y su función de probabilidad de masa.
  """
  probs = [pmf(i) for i in values]
  return lambda: inverse_trans_arr(probs, values)

def inverse_trans_arr(probs, values):
  """
  Una variable aleatoria discreta de valores finitos.

  Generada usando el método de la transformada inversa dado que se conocen
  los valores de su imagen y sus respectivas probabilidades.
  """
  i = 0
  F = probs[0]
  U = random()

  while U >= F:
    i += 1
    F += probs[i]

  return values[i]

def inverse_trans_pmf(initial_val, pmf):
  """
  Una variable aleatoria discreta de valores sucesivos, no necesariamente
  finitos.

  Generada usando el método de la transformada inversa dado que se conoce el
  menor valor de su imagen y su función de probabilidad de masa.
  """
  i = initial_val
  prob_i = pmf(i)
  F = prob_i
  U = random()

  while U >= F:
    i += 1
    prob_i = pmf(i)
    F += prob_i

  return i

def inverse_trans_rec(initial_val, rec_pmf):
  """
  Una variable aleatoria discreta de valores sucesivos, no necesariamente
  finitos.

  Generada usando el método de la transformada inversa dado que se conoce el
  menor valor de su imagen y su función de probabilidad de masa, que sigue una
  definición recursiva.
  """
  i = initial_val
  prob_i = rec_pmf(i)
  F = prob_i
  U = random()

  while U >= F:
    i += 1
    prob_i = rec_pmf(prob_i, i)
    F += prob_i

  return i

def binomial_rec(n, p):
  return inverse_trans_rec(0, (1-p)**n, lambda prev_prob, i: prev_prob * (n-i-1) * p / (i * (1-p)))

def negative_binomial_rec(r, p):
  return inverse_trans_rec(r, p**r, lambda prev_prob, i: prev_prob * (r+i-1) * (1-p) / i)

def poisson_rec(lambd):
  return inverse_trans_rec(0, math.exp(-lambd), lambda prob_prev, i: prob_prev * lambd / i)

def geometric_rec(p):
  return inverse_trans_rec(1, p, lambda prev_prob, _: prev_prob * (1-p))

def hypergeometric_rec(n, N, M):
  return inverse_trans_rec(
    0,
    math.comb(M-N, n) / math.comb(M, n),
    lambda prob_prev, i: prob_prev * (N-(i-1)) * (n-(i-1)) / (i * (i + M - N - n))
  )

def accept_reject(random_var_Y, probs_X, probs_Y, c):
  """
  Una variable aleatoria discreta de valores finitos.

  Generada usando el método de aceptación y rechazo dado que se sabe generar
  otra variable aleatoria discreta finita, y que se conocen las probabilidades
  de cada uno de los valores de ambas variables junto una constante `c` tal
  que `probs_X[i] / probs_Y[i] >= c` para todo `i`.
  """
  Y = random_var_Y()

  while  random() >= probs_X[Y] / (c * probs_Y[Y]):
    Y = random_var_Y()

  return Y

def accept_reject_fun(Y, pmf_X, pmf_Y, values_X, values_Y):
  """
  Produce una función que genera una variable aleatoria discreta de valores
  finitos.

  Generada usando el método de de aceptación y rechazo dado que se sabe generar
  otra variable aleatoria discreta finita, y que se conocen tanto los valores
  como las funciones de probabilidad de masa de ambas variables.
  """
  probs_Y = [pmf_Y(i) for i in values_Y]
  probs_X = [pmf_X(i) for i in values_Y]

  c = max(pmf_X(i) / pmf_Y(i) for i in values_X)

  return lambda: accept_reject(Y, probs_X, probs_Y, c)

def urn(probs, values, n_decimals=2):
  """
  Una variable aleatoria discreta de valores finitos.

  Generada usando el método de la urna, dado que se conocen los posibles
  valores de la variable junto a sus respectivas probabilidades, así como
  la máxima cantidad de decimales que pueden poseer estas últimas.
  """
  # number of elements in urn
  n = 10**n_decimals

  # iterator of lists
  # each list contains an index, reapeated prob(index) * n times
  indices = ([index] * int(prob * n) for index, prob in enumerate(probs))

  # flat index list
  indices = reduce(iconcat, indices, [])

  index = indices[randint(n) - 1]
  return values[index]

