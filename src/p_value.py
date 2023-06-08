import scipy.stats as sp

def statistic(prob, freq):
  size = sum(freq)
  return sum((freq_i - size * prob_i)**2 / (size * prob_i)
             for prob_i, freq_i in zip(prob, freq))

def pearson(prob, freq):
  k = len(freq)
  T = statistic(prob, freq)
  p_val = 1 - sp.chi2.cdf(T, k-1)
  return round(p_val, 4)

def simulate(sims, prob, freq):
  size = sum(freq)
  T = statistic(prob, freq)

  successes = 0
  for _ in range(sims):
    freq = []
    prob_prev = prob
    prob = []
    for prov_prev_i in prob_prev:
      freq_i = size - sum(freq)
      prov_i = prov_prev_i / (1 - sum(prob))
      prob.append(prov_prev_i)
      freq.append(sp.binom.rvs(freq_i, prov_i))
      t = statistic(prob, freq)
    successes += t >= T

  p_val = successes / sims
  return round(p_val, 4)

def statistic_ks(sample, cdf):
  size = len(sample)
  sample.sort()
  values = map(cdf, sample)
  return max(max((j+1)/size - val, val - j/size) for j, val in enumerate(values))

def kolmogorov_smirnov(sims, sample, cdf):
  size = len(sample)
  D = statistic_ks(sample, cdf)

  successes = 0
  for _ in range(sims):
    U_sample = sp.uniform.rvs(size=size)
    d = statistic_ks(U_sample, sp.uniform.cdf)
    successes += d >= D

  p_value = successes / sims
  return round(p_value, 4)
