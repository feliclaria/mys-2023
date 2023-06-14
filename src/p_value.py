import numpy as np
from scipy.stats import uniform, chi2, binom


def pearson_statistic(probs, freqs):
  size = np.sum(freqs)
  return np.sum((freqs - size * probs)**2 / (size * probs))

def pearson_chi2(probs, freqs, params=0, digits=4):
  k = len(freqs)
  t = pearson_statistic(probs, freqs)
  p_val = 1 - chi2.cdf(t, k-params-1)
  return round(p_val, digits)

def pearson_simulate(sims, probs, freqs, digits=4):
  k = len(probs)
  size = np.sum(freqs)
  t = pearson_statistic(probs, freqs)

  successes = 0
  for _ in range(sims):
    freqs_sim = np.zeros(k, dtype=int)
    for j in range(k):
      n = size - np.sum(freqs_sim)
      p = probs[j] / (1 - np.sum(probs[:j]))
      freqs_sim[j] = binom(n=n, p=p).rvs()

    t_sim = pearson_statistic(probs, freqs_sim)
    successes += t <= t_sim
  p_val = successes / sims

  return round(p_val, digits)


def kolmogorov_smirnov_statistic(sample, cdf):
  size = len(sample)
  sample.sort()
  values = map(cdf, sample)
  return max(max((j+1)/size - val, val - j/size) for j, val in enumerate(values))

def kolmogorov_smirnov_simulate(sims, sample, cdf, digits=4):
  size = len(sample)
  d = kolmogorov_smirnov_statistic(sample, cdf)

  successes = 0
  for _ in range(sims):
    sample_sim = uniform.rvs(size=size)
    sample_sim.sort()
    d_sim = kolmogorov_smirnov_statistic(sample_sim, uniform.cdf)
    successes += d <= d_sim
  p_value = successes / sims

  return round(p_value, digits)
