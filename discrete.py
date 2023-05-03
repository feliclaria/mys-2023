from random import random
import math

# TODO: Crear generador de funciones de trans. inversa y aceptación-rechazo

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

def negative_binomial(s, p):
  i = 0
  prob_i = p**s
  F = p
  U = random()

  while F <= U:
    i += 1
    prob_i *= (s+i-1) * (1-p) / i
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
  i = 0
  prob_i = p
  F = prob_i
  U = random()

  while F <= U:
    i += 1
    prob_i *= 1-p
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

def inverse_trans_rec(prob_base, prob_fun):
  i = 0
  prob_i = prob_base
  F = prob_i
  U = random()

  while F <= U:
    i += 1
    prob_i = prob_fun(prob_i, i)
    F += prob_i

  return i

def binomial_rec(n, p):
  return inverse_trans_rec((1-p)**n, lambda prob_prev, i: prob_prev * (n-i-1) * p / (i * (1-p)))

def negative_binomial_rec(s, p):
  return inverse_trans_rec(p**s, lambda prob_prev, i: prob_prev * (s+i-1) * (1-p) / i)

def poisson_rec(lambd):
  return inverse_trans_rec(math.exp(-lambd), lambda prob_prev, i: prob_prev * lambd / i)

def geometric_rec(p):
  return inverse_trans_rec(p, lambda prob_prev, _: prob_prev * (1-p))

def hypergeometric_rec(n, N, M):
  return inverse_trans_rec(
    math.comb(M-N, n) / math.comb(M, n),
    lambda prob_prev, i: prob_prev * (N-(i-1)) * (n-(i-1)) / (i * (i + M - N - n))
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
