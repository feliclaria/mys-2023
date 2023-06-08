from prettytable import PrettyTable
from collections import Counter
import scipy.stats as sp
import p_value as pval
import math


def ex1():
  prob = [0.25, 0.5, 0.25]
  freq = [141, 291, 132]
  sims = 10_000

  pearson = pval.pearson(prob, freq)
  sim = pval.simulate(sims, prob, freq)

  table = PrettyTable(['', 'Pearson', f'{sims} sims.'])
  table.title = 'Ejercicio 1'
  table.add_row(['p-valor', pearson, sim])
  print(table)


def ex2():
  prob = [1/6]*6
  freq = [158, 172, 164, 181, 160, 165]
  sims = 10_000

  pearson = pval.pearson(prob, freq)
  sim = pval.simulate(sims, prob, freq)

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


# def ex5():
#   sample = [6, 7, 3, 4, 7, 3, 7, 2, 6, 3, 7, 8, 2, 1, 3, 5, 8, 7]
#   size = len(sample)
#   n = 8

#   mean = sum(sample) / size
#   p = mean * n

#   counts = Counter(sample).most_common()
#   counts.sort()

#   freq = (count for _, count in counts)
#   prob = (sp.binom.pmf(k, n, p) for k in range(n+1))

#   T = pval.statistic(prob, freq)
#   p_val = 1 - sp.chi2.cdf(T, n-1-1)
#   print(p_val)

# def ex5p():
#   sample = [6, 7, 3, 4, 7, 3, 7, 2, 6, 3, 7, 8, 2, 1, 3, 5, 8, 7]
#   size = len(sample)
#   n = 8

#   mean = sum(sample) / size
#   p = mean * n

#   counts = Counter(sample).most_common()
#   counts.sort()

#   freq = (count for _, count in counts)
#   prob = (sp.binom.pmf(k, n, p) for k in range(n+1))

#   T = pval.statistic(prob, freq)
#   successes = 0

#   sims = 10_000
#   for _ in range(sims):
#     sample = sp.binom.rvs(n, p, size=size)

#     mean = sum(sample) / size
#     pp = mean * n

#     counts = Counter(sample).most_common()
#     counts.sort()

#     freq = (count for _, count in counts)
#     prob = (sp.binom.pmf(k, n, pp) for k in range(n+1))

#     t = pval.statistic(prob, freq)
#     successes += t >= T

#   print(round(successes/sims, 4))

# def sim_binomial(Nsim = 1000):
#     freq = [0, 1, 2, 4, 1, 1, 2, 5, 2]
#     m = sum(freq)
#     n = 8
#     p = sum(i * freq[i] for i in range(9)) / sum(freq) / n
#     rv = sp.binom(n, p)
#     T = sum((freq[i] - m * rv.pmf(i))**2 / (m * rv.pmf(i)) for i in range(9))
#     pvalue = 1 - sp.chi2.cdf(T, 7)


#     rv = sp.binom(8, 0.62)
#     pvalue = 0
#     for _ in range(Nsim):
#         freq = [0] * 9

#         # Genero 18 datos
#         for j in range(18):
#             freq[rv.rvs()] += 1

#         # Calculo p_i(sim)
#         p = sum(i * freq[i] for i in range(9)) / 18 / 8
#         rv2 = sp.binom(8, p)

#         # Calculo el estadistico
#         t = sum((freq[i] - m * rv2.pmf(i))**2 / (m * rv2.pmf(i)) for i in range(9))
#         if t >= T:
#             pvalue += 1

#     return pvalue / Nsim



def main():
  ex1()
  ex2()
  ex3()
  ex4()


if __name__ == '__main__':
  main()