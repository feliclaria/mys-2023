import numpy as np
from collections import Counter
from scipy.stats import uniform, chi2

def sample_counts(sample, bins, values, pmf):
  freqs, _ = np.histogram(sample, bins)
  probs = np.array([pmf(k, sample) for k in values])

  return probs, freqs

def pearson_statistic(probs, freqs):
  size = np.sum(freqs)
  return np.sum((freqs - size * probs)**2 / (size * probs))

def pearson_chi2(probs, freqs, params=0, digits=4):
  k = len(freqs)
  T = pearson_statistic(probs, freqs)
  p_val = 1 - chi2.cdf(T, k-params-1)
  return round(p_val, digits)

def pearson_simulate(sims, probs, freqs, bins, values, rvs, pmf, digits=4):
  size = np.sum(freqs)
  t = pearson_statistic(probs, freqs)

  successes = 0
  for _ in range(sims):
    sample_sim = rvs(size)
    probs_sim, freqs_sim = sample_counts(sample_sim, bins, values, pmf)
    t_sim = pearson_statistic(probs_sim, freqs_sim)
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
