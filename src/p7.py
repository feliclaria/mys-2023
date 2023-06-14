from prettytable import PrettyTable
from collections import Counter
import scipy.stats as sp
import numpy as np
import p_value as pval

SIMS = 10_000


def ex1():
  probs = np.array([0.25, 0.5, 0.25])
  freqs = np.array([141, 291, 132])

  p_value_chi2 = pval.pearson_chi2(probs, freqs)
  p_value_binom = pval.pearson_simulate(SIMS, probs, freqs)

  table = PrettyTable(['', 'Chi cuadrado', f'{SIMS} sims.'])
  table.title = 'Ejercicio 1'
  table.add_row(['p-valor', p_value_chi2, p_value_binom])
  print(table)


def ex2():
  probs = np.array([1/6]*6)
  freqs = np.array([158, 172, 164, 181, 160, 165])

  p_value_chi2 = pval.pearson_chi2(probs, freqs)
  p_value_sims = pval.pearson_simulate(SIMS, probs, freqs)

  table = PrettyTable(['', 'Chi cuadrado', f'{SIMS} sims.'])
  table.title = 'Ejercicio 2'
  table.add_row(['p-valor', p_value_chi2, p_value_sims])
  print(table)


def ex3():
  sample = [0.12, 0.18, 0.06, 0.33, 0.72, 0.83, 0.36, 0.27, 0.77, 0.74]
  cdf = sp.uniform.cdf
  p_value_sims = pval.kolmogorov_smirnov_simulate(SIMS, sample, cdf)

  table = PrettyTable(['Nro. sims.', 'p-valor'])
  table.title = 'Ejercicio 3'
  table.add_row([SIMS, p_value_sims])
  print(table)


def ex4():
  sample = [
    86.0, 133.0, 75.0, 22.0, 11.0, 144.0, 78.0,
    122.0, 8.0, 146.0, 33.0, 41.0, 99.0
  ]
  cdf = sp.expon(scale=50).cdf
  p_value_sims = pval.kolmogorov_smirnov_simulate(SIMS, sample, cdf)

  table = PrettyTable(['Nro. sims.', 'p-valor'])
  table.title = 'Ejercicio 4'
  table.add_row([SIMS, p_value_sims])
  print(table)


def ex5_old_chi2(probs, freqs):
  return pval.pearson_chi2(probs, freqs, params=1, digits=10)

def ex5_old_sims(sims, probs, freqs, dist, support):
  size = np.sum(freqs)
  t = pval.pearson_statistic(probs, freqs)

  successes = 0
  for _ in range(sims):
    sample_sim = dist.rvs(size)

    p_sim = np.mean(sample_sim) / 8
    dist_sim = sp.binom(8, p_sim)

    probs_sim, freqs_sim = pval.group_sample(sample_sim, dist_sim.pmf, support)
    t_sim = pval.pearson_statistic(probs_sim, freqs_sim)
    successes += t <= t_sim

  return successes / sims

def ex5_old():
  sample = np.array([
    6, 7, 3, 4, 7, 3, 7, 2, 6,
    3, 7, 8, 2, 1, 3, 5, 8, 7
  ])

  n = 8
  p =  np.mean(sample) / n
  dist = sp.binom(n, p)

  support = range(n+1)
  counts = Counter(sample)

  probs = np.array([dist.pmf(k) for k in support])
  freqs = np.array([counts.get(k, 0) for k in support])

  p_value_chi2 = ex5_old_chi2(probs, freqs)
  p_value_sims = ex5_old_sims(SIMS, probs, freqs, dist, support)

  table = PrettyTable(['', 'Pearson', f'{SIMS} sims.'])
  table.title = 'Ejercicio 5'
  table.add_row(['p-valor', p_value_chi2, p_value_sims])
  print(table)

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

  table = PrettyTable(['', 'Pearson', f'{SIMS} sims.'])
  table.title = 'Ejercicio 5'
  table.add_row(['p-valor', p_value_chi2, p_value_sims])
  print(table)


def ex6():
  probs = np.array([31, 22, 12, 10, 8, 6, 4, 4, 2, 1])
  freqs = np.array([188, 138, 87, 65, 48, 32, 30, 34, 13, 2])

  sample_table = PrettyTable(['Premio', 'Prob', 'Freq'])
  sample_table.title = 'Ej. 6: Datos'
  for i, (prob, freq) in enumerate(zip(probs, freqs)):
    sample_table.add_row([i+1, f'{prob}%', freq])
  print(sample_table)

  probs = probs / 100

  p_value_chi2 = pval.pearson_chi2(probs, freqs)
  p_value_sims = pval.pearson_simulate(SIMS, probs, freqs)

  results_table = PrettyTable(['Método', 'p-valor'])
  results_table.title = 'Ej. 6: Resultados'
  results_table.add_row(['Chi-cuadrado', p_value_chi2])
  results_table.add_row([f'{SIMS} sims.', p_value_sims])
  print(results_table)


def ex7():
  size = 30
  dist = sp.expon
  sample = dist.rvs(size=size)

  p_value = pval.kolmogorov_smirnov_simulate(SIMS, sample, dist.cdf)

  table = PrettyTable(['Método', 'p-valor'])
  table.title = 'Ejercicio 7'
  table.add_row([f'{SIMS} sims.', p_value])
  print(table)


def main():
  ex1()
  ex2()

if __name__ == '__main__':
  main()