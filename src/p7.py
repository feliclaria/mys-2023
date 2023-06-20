from continuous import rt
from tabulate import tabulate
from sys import argv

import p_value as pval
import scipy.stats as sp
import numpy as np
import math


SIMS = 10_000

def tabulate_pearson(t, p_value_chi2, p_value_sims):
  table = [
    ['method', 't', 'p-value'],
    ['chi-squared', t, p_value_chi2],
    [f'{SIMS} sims', t, p_value_sims]
  ]
  return tabulate(table, headers='firstrow')

def tabulate_kolmogorov_smirnov(d, p_value):
  table = [
    ['method', 'd', 'p-value'],
    [f'{SIMS} sims', d, p_value]
  ]
  return tabulate(table, headers='firstrow')


def ex1():
  probs = np.array([0.25, 0.5, 0.25])
  freqs = np.array([141, 291, 132])

  t, p_value_chi2 = pval.pearson_chi2(probs, freqs)
  p_value_sims = pval.pearson_sims(SIMS, probs, freqs)
  print(tabulate_pearson(t, p_value_chi2, p_value_sims))


def ex2():
  probs = np.array([1/6]*6)
  freqs = np.array([158, 172, 164, 181, 160, 165])

  t, p_value_chi2 = pval.pearson_chi2(probs, freqs)
  p_value_sims = pval.pearson_sims(SIMS, probs, freqs)
  print(tabulate_pearson(t, p_value_chi2, p_value_sims))


def ex3():
  sample = [0.12, 0.18, 0.06, 0.33, 0.72, 0.83, 0.36, 0.27, 0.77, 0.74]
  cdf = sp.uniform.cdf

  d, p_value = pval.kolmogorov_smirnov_sims(SIMS, sample, cdf)
  print(tabulate_kolmogorov_smirnov(d, p_value))


def ex4():
  sample = [
    86.0, 133.0, 75.0, 22.0, 11.0, 144.0, 78.0,
    122.0, 8.0, 146.0, 33.0, 41.0, 99.0
  ]
  cdf = sp.expon(scale=50).cdf

  d, p_value = pval.kolmogorov_smirnov_sims(SIMS, sample, cdf)
  print(tabulate_kolmogorov_smirnov(d, p_value))


def ex5():
  sample = np.array([
    6, 7, 3, 4, 7, 3, 7, 2, 6,
    3, 7, 8, 2, 1, 3, 5, 8, 7
  ])

  n = 8
  support = range(n+1)
  dist_estimator = lambda sample: sp.binom(n, np.mean(sample) / n)

  t, p_value_chi2 = pval.pearson_chi2_estimate(sample, dist_estimator, support, params=1, digits=10)
  p_value_sims = pval.pearson_sims_estimate(SIMS, sample, dist_estimator, support, digits=10)
  print(tabulate_pearson(t, p_value_chi2, p_value_sims))


def ex6():
  probs = np.array([31, 22, 12, 10, 8, 6, 4, 4, 2, 1])
  freqs = np.array([188, 138, 87, 65, 48, 32, 30, 34, 13, 2])

  data_table = [['prize', 'prob', 'freq']]
  for i, (prob, freq) in enumerate(zip(probs, freqs)):
    data_table.append([i+1, f'{prob}%', freq])
  print(tabulate(data_table, headers='firstrow', colalign=('center', 'center')))
  print()

  probs = probs / 100

  t, p_value_chi2 = pval.pearson_chi2(probs, freqs)
  p_value_sims = pval.pearson_sims(SIMS, probs, freqs)
  print(tabulate_pearson(t, p_value_chi2, p_value_sims))


def ex7():
  size = 30
  dist = sp.expon
  sample = dist.rvs(size=size)

  d, p_value = pval.kolmogorov_smirnov_sims(SIMS, sample, dist.cdf)
  print(tabulate_kolmogorov_smirnov(d, p_value))


def ex8():
  sizes = [10, 20, 100, 1_000]
  df = 11
  normal_cdf = lambda x: math.erf(x / math.sqrt(2.0)) / 2.0 + 0.5

  results = [['size', 'd', 'p-value']]

  for size in sizes:
    sample = [rt(df) for _ in range(size)]
    d, p_value = pval.kolmogorov_smirnov_sims(SIMS, sample, normal_cdf)
    results.append([size, d, p_value])

  print(tabulate(results, headers='firstrow'))


def ex9():
  sample = np.array([
    1.6, 10.3, 3.5, 13.5, 18.4, 7.7, 24.3,
    10.7, 8.4, 4.9, 7.9, 12, 16.2, 6.8, 14.7
  ])
  dist_estimator = lambda sample: sp.expon(scale=np.mean(sample))

  d, p_value = pval.kolmogorov_smirnov_sims_estimate(SIMS, sample, dist_estimator)
  print(tabulate_kolmogorov_smirnov(d, p_value))


def ex10():
  sample = np.array([
    91.9, 97.8, 111.4, 122.3, 105.4, 95.0,
    103.8, 99.6, 96.6, 119.3, 104.8, 101.7
  ])
  def dist_estimator(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)
    return sp.norm(loc=mu, scale=sigma)

  d, p_value = pval.kolmogorov_smirnov_sims_estimate(SIMS, sample, dist_estimator)
  print(tabulate_kolmogorov_smirnov(d, p_value))


def ex(k, fun):
  print()
  print(tabulate([[f'(Ex. {k})']]))
  print()
  fun()
  print()

def main(argv):
  k = int(argv[1])
  funs = [ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex10]

  if k == 0:
    for i, fun in enumerate(funs): ex(i+1, fun)
  else: ex(k, funs[k-1])

if __name__ == '__main__':
  main(argv)