from prettytable import PrettyTable
from collections import Counter
import scipy.stats as sp
import numpy as np
import p_value as pval

sims = 10_000

def ex1():
  probs = np.array([0.25, 0.5, 0.25])
  freqs = np.array([141, 291, 132])

  p_value_chi2 = pval.pearson_chi2(probs, freqs)
  p_value_sims = pval.pearson_simulate(sims, probs, freqs)

  table = PrettyTable(['', 'Pearson', f'{sims} sims.'])
  table.title = 'Ejercicio 1'
  table.add_row(['p-valor', p_value_chi2, p_value_sims])
  print(table)


def ex2():
  probs = [1/6]*6
  freqs = [158, 172, 164, 181, 160, 165]

  pearson = pval.pearson_chi2(probs, freqs)
  sim = pval.pearson_simulate(sims, probs, freqs)

  table = PrettyTable(['', 'Pearson', f'{sims} sims.'])
  table.title = 'Ejercicio 2'
  table.add_row(['p-valor', pearson, sim])
  print(table)


def ex3():
  sample = [0.12, 0.18, 0.06, 0.33, 0.72, 0.83, 0.36, 0.27, 0.77, 0.74]
  cdf = sp.uniform.cdf
  p_value = pval.kolmogorov_smirnov_simulate(sims, sample, cdf)

  table = PrettyTable(['Nro. sims.', 'p-valor'])
  table.title = 'Ejercicio 3'
  table.add_row([sims, p_value])
  print(table)


def ex4():
  sample = [
    86.0, 133.0, 75.0, 22.0, 11.0, 144.0, 78.0,
    122.0, 8.0, 146.0, 33.0, 41.0, 99.0
  ]
  cdf = lambda x: sp.expon.cdf(x, scale=50)
  p_value = pval.kolmogorov_smirnov_simulate(sims, sample, cdf)

  table = PrettyTable(['Nro. sims.', 'p-valor'])
  table.title = 'Ejercicio 4'
  table.add_row([sims, p_value])
  print(table)


def ex5():
  sample = np.array([
    6, 7, 3, 4, 7, 3, 7, 2, 6,
    3, 7, 8, 2, 1, 3, 5, 8, 7
  ])

  n = 8
  p =  np.mean(sample) / n
  dist = sp.binom(n, p)

  bins = range(n+2)
  values = range(n+1)
  pmf = lambda x, _: dist.pmf(x)

  probs_chi2, freqs_chi2 = pval.sample_counts(sample, bins, values, pmf)
  p_value_chi2 = pval.pearson_chi2(probs_chi2, freqs_chi2, params=1)

  def pmf_sim(k, sample_sim):
    n = 8
    p = np.mean(sample_sim) / n
    return sp.binom(n, p).pmf(k)

  probs_sims, freqs_sims = pval.sample_counts(sample, bins, values, pmf)
  p_value_sims = pval.pearson_simulate(sims, probs_sims, freqs_sims,
                                       bins, values, dist.rvs, pmf_sim)

  table = PrettyTable(['', 'Pearson', f'{sims} sims.'])
  table.title = 'Ejercicio 5'
  table.add_row(['p-valor', p_value_chi2, p_value_sims])
  print(table)




def ex5_sims():
  sample = [6, 7, 3, 4, 7, 3, 7, 2, 6, 3, 7, 8, 2, 1, 3, 5, 8, 7]

  mean = sum(sample) / len(sample)

  n = 8
  p = mean / n
  binom_dist = sp.binom(n, p)

  values = range(n+1)
  probs, freqs = pval.sample_counts(list(sample), values, binom_dist.pmf)

  return pval.pearson_simulate(sims, probs, freqs, values, binom_dist)


def si(*params):
  print(params)

def main():

  ex5()

if __name__ == '__main__':
  main()