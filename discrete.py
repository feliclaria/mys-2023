from random import random
from functools import reduce
from operator import iconcat
import math

def randint(N):
  return int(N * random()) + 1

def binomial(n, p):
  i = 0
  prob_i = (1 - p)**n
  F = prob_i
  U = random()

  while F <= U:
    i += 1
    prob_i *= (n-i-1) * p / (i * (1-p))
    F += prob_i

  return i

def negative_binomial(r, p):
  i = r
  prob_i = p**r
  F = p
  U = random()

  while F <= U:
    i += 1
    prob_i *= (r+i-1) * (1-p) / i
    F += p

  return i

def poisson(lambd):
  i = 0
  prob_i = math.exp(-lambd)
  F = prob_i
  U = random()

  while F <= U:
    i += 1
    prob_i *= lambd / i
    F += prob_i

  return i

def poisson_fast(lambd):
  prob_i = math.exp(-lambd)
  F = prob_i

  for i in range(1, int(lambd)+1):
    prob_i *= lambd / i
    F += prob_i

  U = random()

  if F <= U:
    i = int(lambd) + 1

    while F <= U:
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
  i = 1
  prob_i = p
  F = prob_i
  U = random()

  while F <= U:
    i += 1
    prob_i *= (1-p)
    F += prob_i

  return i

def hypergeometric(n, N, M):
  i = 0
  prob_i = math.comb(M-N, n) / math.comb(M, n)
  F = prob_i
  U = random()

  while F <= U:
    i += 1
    prob_i *= (N-(i-1)) * (n-(i-1)) / (i * (i + M - N - n))
    F += prob_i

  return i

def inverse_trans_arr(probs, values):
  i = 0
  F = probs[0]
  U = random()

  while F <= U:
    i += 1
    F += probs[i]

  return values[i]

def inverse_trans_rec(initial_val, initial_prob, inductive_prob):
  i = initial_val
  prob_i = initial_prob
  F = prob_i
  U = random()

  while F <= U:
    i += 1
    prob_i = inductive_prob(prob_i, i)
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
  Y = random_var_Y()

  while probs_X[Y] / (c * probs_Y[Y]) <= random():
    Y = random_var_Y()

  return Y

def urn(probs, values):
  # iterator of lists
  # each list contains an index, reapeated prob(index) * 100 times
  indices = ([index] * int(prob * 100) for index, prob in enumerate(probs))

  # flat index list
  indices = reduce(iconcat, indices, [])

  index = indices[randint(100) - 1]
  return values[index]
