from random import random
from continuous import exponential

def homogeneous(lambd, T):
  events = []
  N_t = 0
  t = exponential(lambd)
  while t < T:
    events.append(t)
    N_t += 1
    t += exponential(lambd)
  return N_t, events

def inhomogeneous(lambd, lambd_t, T):
  events = []
  t = exponential(lambd)
  N_t = 0
  while t < T:
    if random() < lambd_t(t) / lambd:
      events.append(t)
      N_t += 1
    t += exponential(lambd)
  return N_t, events

def inhomogeneous_improved(lambd, interv, T):
  j = 0
  events = []
  N_t = 0
  t = exponential(lambd[j])
  while t <= T:
    if t <= interv[j]:
      if random() < (2 * t + 1) / lambd[j]:
        events.append(t)
        N_t += 1
      t += exponential(lambd[j])
    else:
      t = interv[j] + (t - interv[j]) * lambd[j] / lambd[j + 1]
      j += 1
  return N_t, events