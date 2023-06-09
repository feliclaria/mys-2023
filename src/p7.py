from prettytable import PrettyTable
from collections import Counter
import scipy.stats as sp
import p_value as pval
import math


def ex1():
  probs = [0.25, 0.5, 0.25]
  freqs = [141, 291, 132]
  sims = 10_000

  pearson = pval.pearson(probs, freqs)
  sim = pval.pearson_sim(sims, probs, freqs)

  table = PrettyTable(['', 'Pearson', f'{sims} sims.'])
  table.title = 'Ejercicio 1'
  table.add_row(['p-valor', pearson, sim])
  print(table)


def ex2():
  probs = [1/6]*6
  freqs = [158, 172, 164, 181, 160, 165]
  sims = 10_000

  pearson = pval.pearson(probs, freqs)
  sim = pval.pearson_sim(sims, probs, freqs)

  table = PrettyTable(['', 'Pearson', f'{sims} sims.'])
  table.title = 'Ejercicio 2'
  table.add_row(['p-valor', pearson, sim])
  print(table)


def ex3():
  sims = 10_000
  sample = [0.12, 0.18, 0.06, 0.33, 0.72, 0.83, 0.36, 0.27, 0.77, 0.74]
  cdf = sp.uniform.cdf
  p_value = pval.kolmogorov_smirnov(sims, sample, cdf)

  table = PrettyTable(['Nro. sims.', 'p-valor'])
  table.title = 'Ejercicio 3'
  table.add_row([sims, p_value])
  print(table)


def ex4():
  sims = 10_000
  sample = [
    86.0, 133.0, 75.0, 22.0, 11.0, 144.0, 78.0,
    122.0, 8.0, 146.0, 33.0, 41.0, 99.0
  ]
  cdf = lambda x: sp.expon.cdf(x, scale=50)
  p_value = pval.kolmogorov_smirnov(sims, sample, cdf)

  table = PrettyTable(['Nro. sims.', 'p-valor'])
  table.title = 'Ejercicio 4'
  table.add_row([sims, p_value])
  print(table)


def ex5_pearson():
  sample = [6, 7, 3, 4, 7, 3, 7, 2, 6, 3, 7, 8, 2, 1, 3, 5, 8, 7]
  mean = sum(sample) / len(sample)

  n = 8
  p = mean / n

  values = range(n+1)
  probs, freqs = pval.stats_disc(sample, values, sp.binom(n, p).pmf)

  return(pval.pearson_estimate(probs, freqs, 1))

def ex5_sims():
  sample = [6, 7, 3, 4, 7, 3, 7, 2, 6, 3, 7, 8, 2, 1, 3, 5, 8, 7]
  size = len(sample)
  mean = sum(sample) / size

  n = 8
  p = mean / n

  values = range(n+1)
  probs, freqs = pval.stats_disc(sample, values, sp.binom(n, p).pmf)

  T = pval.statistic(probs, freqs)

  sims = 10_000
  successes = 0
  for _ in range(sims):
    sample = list(sp.binom(n, p).rvs(size=size))
    mean = sum(sample) / size

    p_sim = mean / n
    probs, freqs = pval.stats_disc(sample, values, sp.binom(n, p_sim).pmf)
    t = pval.statistic(probs, freqs)
    successes += t >= T

  p_val = successes / sims
  return round(p_val, 10)


def main():
  ex1()
if __name__ == '__main__':
  main()