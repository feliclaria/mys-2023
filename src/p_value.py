import scipy.stats as sp

def stats_disc(sample, values, pmf):
  probs = [pmf(k) for k in values]
  freqs = [sample.count(k) for k in values]
  return probs, freqs

def statistic(probs, freqs):
  size = sum(freqs)
  return sum((freq - size * prob)**2 / (size * prob)
              for prob, freq in zip(probs, freqs))

def pearson(probs, freqs, digits=4):
  k = len(freqs)
  T = statistic(probs, freqs)
  p_val = 1 - sp.chi2.cdf(T, k-1)
  return round(p_val, digits)

def pearson_estimate(probs, freqs, m, digits=4):
  k = len(freqs)
  T = statistic(probs, freqs)
  p_val = 1 - sp.chi2.cdf(T, k-m-1)
  return round(p_val, digits)

def pearson_sim(sims, probs, freqs, digits=4):
  size = sum(freqs)
  T = statistic(probs, freqs)

  successes = 0
  for _ in range(sims):
    freqs = []
    prev_probs = probs
    probs = []
    for pp in prev_probs:
      n = size - sum(freqs)
      p = pp / (1 - sum(probs))
      probs.append(pp)
      freqs.append(sp.binom.rvs(n, p))
      t = statistic(probs, freqs)
    successes += t >= T

  p_val = successes / sims
  return round(p_val, digits)

def statistic_ks(sample, cdf):
  size = len(sample)
  sample.sort()
  values = map(cdf, sample)
  return max(max((j+1)/size - val, val - j/size) for j, val in enumerate(values))

def kolmogorov_smirnov(sims, sample, cdf, digits=4):
  size = len(sample)
  D = statistic_ks(sample, cdf)

  successes = 0
  for _ in range(sims):
    U_sample = sp.uniform.rvs(size=size)
    d = statistic_ks(U_sample, sp.uniform.cdf)
    successes += d >= D

  p_value = successes / sims
  return round(p_value, digits)
