import scipy.stats as sp
import numpy as np
import p_value as pval

from sys import argv
from tabulate import tabulate


SIMS = 10_000


def tabulate_pearson(p_value_chi2, p_value_sims):
  table = [
      ['method', 'p-value'],
      ['chi-squared', p_value_chi2],
      [f'{SIMS} sims', p_value_sims]
    ]
  return tabulate(table, headers='firstrow')

def tabulate_kolmogorov_smirnov(p_value_sims):
  table = [[f'{SIMS} sims', p_value_sims]]
  return tabulate(table)


def ex1():
  probs = np.array([0.25, 0.5, 0.25])
  freqs = np.array([141, 291, 132])

  p_value_chi2 = pval.pearson_chi2(probs, freqs)
  p_value_sims = pval.pearson_simulate(SIMS, probs, freqs)
  print(tabulate_pearson(p_value_chi2, p_value_sims))


def ex2():
  probs = np.array([1/6]*6)
  freqs = np.array([158, 172, 164, 181, 160, 165])

  p_value_chi2 = pval.pearson_chi2(probs, freqs)
  p_value_sims = pval.pearson_simulate(SIMS, probs, freqs)
  print(tabulate_pearson(p_value_chi2, p_value_sims))


def ex3():
  sample = [0.12, 0.18, 0.06, 0.33, 0.72, 0.83, 0.36, 0.27, 0.77, 0.74]
  cdf = sp.uniform.cdf

  p_value_sims = pval.kolmogorov_smirnov_simulate(SIMS, sample, cdf)
  print(tabulate_kolmogorov_smirnov(p_value_sims))


def ex4():
  sample = [
    86.0, 133.0, 75.0, 22.0, 11.0, 144.0, 78.0,
    122.0, 8.0, 146.0, 33.0, 41.0, 99.0
  ]
  cdf = sp.expon(scale=50).cdf

  p_value_sims = pval.kolmogorov_smirnov_simulate(SIMS, sample, cdf)
  print(tabulate_kolmogorov_smirnov(p_value_sims))


def ex5():
  sample = np.array([
    6, 7, 3, 4, 7, 3, 7, 2, 6,
    3, 7, 8, 2, 1, 3, 5, 8, 7
  ])

  n = 8
  support = range(n+1)
  dist_estimator = lambda sample: sp.binom(n, np.mean(sample) / n)

  p_value_chi2 = pval.pearson_chi2_from_sample(sample, dist_estimator, support, params=1, digits=10)
  p_value_sims = pval.pearson_simulate_from_sample(SIMS, sample, dist_estimator, support, digits=10)
  print(tabulate_pearson(p_value_chi2, p_value_sims))


def ex6():
  probs = np.array([31, 22, 12, 10, 8, 6, 4, 4, 2, 1])
  freqs = np.array([188, 138, 87, 65, 48, 32, 30, 34, 13, 2])

  data_table = [['prize', 'prob', 'freq']]
  for i, (prob, freq) in enumerate(zip(probs, freqs)):
    data_table.append([i+1, f'{prob}%', freq])
  print(tabulate(data_table, headers='firstrow', colalign=('center', 'center')))
  print()

  probs = probs / 100

  p_value_chi2 = pval.pearson_chi2(probs, freqs)
  p_value_sims = pval.pearson_simulate(SIMS, probs, freqs)
  print(tabulate_pearson(p_value_chi2, p_value_sims))


def ex7():
  size = 30
  dist = sp.expon
  sample = dist.rvs(size=size)

  p_value_sims = pval.kolmogorov_smirnov_simulate(SIMS, sample, dist.cdf)
  print(tabulate_kolmogorov_smirnov(p_value_sims))


def main(argv):
  k = int(argv[1])
  funs = [ex1, ex2, ex3, ex4, ex5, ex6, ex7]

  if k == 0:
    for i, f in enumerate(funs):
      print()
      print(f'+--- (Ex. {i+1}) ---+')
      f()
  else:
    print(f'+--- (Ex. {k}) ---+')
    funs[k-1]()

if __name__ == '__main__':
  main(argv)