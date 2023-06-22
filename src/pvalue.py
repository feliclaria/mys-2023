import numpy as np
from random import random
from scipy.stats import chi2, binom
from collections import Counter


def group_sample(sample, pmf, support):
  counts = Counter(sample)
  freqs = np.array([counts.get(k, 0) for k in support])
  probs = np.array([pmf(k) for k in support])
  return probs, freqs


def pearson_statistic(probs, freqs):
  size = np.sum(freqs)
  return np.sum((freqs - size * probs)**2 / (size * probs))

def pearson_chi2(probs, freqs, params=0, digits=None):
  k = len(freqs)
  t = pearson_statistic(probs, freqs)
  p_value = 1 - chi2.cdf(t, k-params-1)

  if digits is not None:
    p_value = round(p_value, digits)
  return t, p_value

def pearson_chi2_estimate(sample, dist_estimator, support, params=0, digits=None):
  dist = dist_estimator(sample)
  probs, freqs = group_sample(sample, dist.pmf, support)

  t, p_value = pearson_chi2(probs, freqs, params, digits)

  if digits is not None:
    t = round(t, digits)
    p_value = round(p_value, digits)
  return t, p_value

def pearson_sims(sims, probs, freqs, digits=None):
  k = len(probs)
  size = np.sum(freqs)

  t = pearson_statistic(probs, freqs)

  p = np.zeros(k, dtype=float)
  for j in range(k):
    p[j] = probs[j] / (1 - np.sum(probs[:j]))
    if p[j] > 1: p[j] = 1

  successes = 0
  for _ in range(sims):
    freqs_sim = np.zeros(k, dtype=int)
    for j in range(k):
      n = size - np.sum(freqs_sim)
      freqs_sim[j] = binom(n, p[j]).rvs()

    t_sim = pearson_statistic(probs, freqs_sim)
    successes += t <= t_sim
  p_value = successes / sims

  if digits is not None:
    p_value = round(p_value, digits)
  return p_value

def pearson_sims_estimate(sims, sample, dist_estimator, support, digits=None):
  size = len(sample)
  dist = dist_estimator(sample)
  probs, freqs = group_sample(sample, dist.pmf, support)
  t = pearson_statistic(probs, freqs)

  successes = 0
  for _ in range(sims):
    sample_sim = dist.rvs(size)
    dist_sim = dist_estimator(sample_sim)
    probs_sim, freqs_sim = group_sample(sample_sim, dist_sim.pmf, support)
    t_sim = pearson_statistic(probs_sim, freqs_sim)
    successes += t <= t_sim
  p_value = successes / sims

  if digits is not None:
    p_value = round(p_value, digits)
  return p_value


def kolmogorov_smirnov_statistic(sample, cdf):
  size = len(sample)
  sample.sort()
  values = map(cdf, sample)
  return max(max((j+1)/size - val, val - j/size) for j, val in enumerate(values))

def kolmogorov_smirnov_sims(sims, sample, cdf, digits=None):
  size = len(sample)
  d = kolmogorov_smirnov_statistic(sample, cdf)

  successes = 0
  for _ in range(sims):
    sample_sim = [random() for _ in range(size)]
    d_sim = kolmogorov_smirnov_statistic(sample_sim, lambda x: x)
    successes += d <= d_sim
  p_value = successes / sims

  if digits is not None:
    p_value = round(p_value, digits)
  return d, p_value

def kolmogorov_smirnov_sims_estimate(sims, sample, dist_estimator, digits=None):
  size = len(sample)
  dist = dist_estimator(sample)
  d = kolmogorov_smirnov_statistic(sample, dist.cdf)

  successes = 0
  for _ in range(sims):
    sample_sim = dist.rvs(size=size)
    dist_sim = dist_estimator(sample_sim)
    d_sim = kolmogorov_smirnov_statistic(sample_sim, dist_sim.cdf)
    successes += d <= d_sim
  p_value = successes / sims

  if digits is not None:
    p_value = round(p_value, digits)
  return d, p_value
