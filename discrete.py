from random import random
import math

def randint(N):
  return int(N * random()) + 1

def binomial(n, p):
  i = 0
  F = (1 - p)**n
  U = random()

  while F <= U:
    F += F * (n-i) * p / ((i+1) * (1-p))
    i += 1

  return i

def negative_binomial(s, p):
  i = 0
  F = p**s
  U = random()

  while F <= U:
    F += F * (s+i) * (1-p) / (i+1)
    i += 1

  return i

def poisson(lambd):
  i = 0
  F = math.exp(-lambd)
  U = random()

  while F <= U:
    F += F * lambd / (i+1)
    i += 1

  return i

def poisson_optimized(lambd):
  F = math.exp(-lambd)

  for j in range(int(lambd)):
    F += F * lambd / (j+1)

  U = random()

  if F <= U:
    j = int(lambd) + 1

    while F <= U:
      F += F * lambd / j
      j += 1

    return j - 1

  else:
    j = int(lambd)

    while U < F:
      F -= F * j / lambd
      j -= 1

    return j + 1

def geometric(p):
  i = 0
  F = p
  U = random()

  while F <= U:
    F += F * (1-p)
    i += 1

  return i

def hypergeometric(n, N, M):
  i = 0
  F = math.comb(M-N, n) / math.comb(M, n)
  U = random()

  while F <= U:
    F += F * (N-i) * (n-i) / ((i+1) * (1 - N + M - n + i))
    i += 1

  return i

def inverse_trans_arr(probs, values):
  i = 0
  F = probs[0]
  U = random()

  while F <= U:
    i += 1
    F += probs[i]

  return values[i]

def inverse_trans_rec(prob_base, prob_next):
  i = 0
  F = prob_base
  U = random()

  while F <= U:
    F += prob_next(F, i)
    i += 1

  return i

def binomial_rec(n, p):
  return inverse_trans_rec((1-p)**n, lambda prob, i: prob * (n-i) * p / ((i+1) * (1-p)))

def negative_binomial_rec(s, p):
  return inverse_trans_rec(p**s, lambda prob, i: prob * (s+i) * (1-p) / (i+1))

def poisson_rec(lambd):
  return inverse_trans_rec(math.exp(-lambd), lambda prob, i: prob * lambd / (i+1))

def geometric_rec(p):
  return inverse_trans_rec(p, lambda prob, _: prob * (1-p))

def hypergeometric_rec(n, N, M):
  return inverse_trans_rec(
    math.comb(M-N, n) / math.comb(M, n),
    lambda prob, i: prob * (N-i) * (n-i) / ((i+1) * (1 + i + M - N - n))
  )

def accept_reject(random_var_Y, probs_X, probs_Y, c):
  Y = random_var_Y()

  while probs_X[Y] / (c * probs_Y[Y]) <= random():
    Y = random_var_Y()

  return Y

def urn_random(probs, values):
  A = []

  for index, prob in enumerate(probs):
    for _ in range(int(prob * 100)):
      A.append(values[index])

  return A[randint(100) - 1]