from pvalue import pearson_statistic, pearson_chi2, pearson_sims
from scipy.stats import poisson, binom
from numpy import array

poisson_dist = poisson(mu=3)

def rvs(size):
  sample = []
  for _ in range(size):
    x = poisson_dist.rvs()
    if x <= 5: sample.append(x)
    else: sample.append(6)
  return sample

def pmf(k):
  if k <= 5: return poisson_dist.pmf(k)
  else: return 1 - poisson_dist.cdf(5)

def frequencies(sample):
  freqs = [0]*7
  for x in sample:
    if x <= 5: freqs[x] += 1
    else: freqs[6] += 1
  return array(freqs)

def pvalue_pearson_sims(sims, probs, freqs):
  size = sum(freqs)
  t = pearson_statistic(probs, freqs)

  successes = 0
  for _ in range(sims):
    sample_sim = rvs(size)
    freqs_sim = frequencies(sample_sim)
    t_sim = pearson_statistic(probs, freqs_sim)
    successes += t <= t_sim
  p_value = successes / sims

  return p_value

def main():
  sample = [4, 3, 3, 3, 1, 2, 6, 2, 1, 3, 1, 2, 1, 0, 4, 4, 5, 1, 5, 2]
  sims = 10_000

  probs = array([pmf(k) for k in range(7)])
  freqs = frequencies(sample)

  t, p_chi2 = pearson_chi2(probs, freqs)
  print(t)
  print(p_chi2)

  p_sims_bin = pearson_sims(sims, probs, freqs)
  print(p_sims_bin)

  p_sims = pvalue_pearson_sims(sims, probs, freqs)
  print(p_sims)


if __name__ == '__main__':
  main()


