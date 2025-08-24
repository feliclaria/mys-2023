from pvalue import pearson_chi2, pearson_sims
from pprocess import inhomogeneous, inhomogeneous_improved
from simulate import mean
from estimate import rate_interval
from random import random
from tabulate import tabulate
from numpy import array
from math import sqrt

SIMS = 10_000

def ejercicio1():
  print()
  print('--- Ejercicio 1 ---')
  print()

  colores = [
    'blanco', 'plateado', 'negro', 'gris', 'rojo',
    'marrón', 'azul', 'verde', 'otros'
  ]
  freqs = array([120, 114, 92, 85, 34, 33, 45, 11, 5])
  probs = array([0.22, 0.20, 0.19, 0.12, 0.09, 0.08, 0.07, 0.02, 0.01])

  data = [['i', 'color', 'frecuencia', 'prob. global']]
  for color, freq, prob in zip(colores, freqs, probs):
    data.append([color, freq, prob])
  print(tabulate(data, headers='firstrow', showindex=True))
  print()

  t, pvalue_chi2 = pearson_chi2(probs, freqs)
  pvalue_sims = pearson_sims(SIMS, probs, freqs)

  results = [
    ['estadístico', t],
    ['p-valor chi-2', pvalue_chi2],
    ['p-valor sims.', pvalue_sims]
  ]
  print(tabulate(results))


def ejercicio2a(T):
  lambd_t = lambda t: (t-3)**2 if 0 <= t <= 6 else 0
  lambd = 9 # en t=0 y t=6
  return inhomogeneous(lambd, lambd_t, T)

def ejercicio2b(T):
  interv = [1.5, 3, 4.5, 6]
  lambd = [9, 2.25, 2.25, 9] # en t = 0, 1.5, 4.5 y 6 respectivamente
  return inhomogeneous_improved(lambd, interv, T)

def ejercicio2():
  print()
  print('--- Ejercicio 2 ---')
  print()

  t = 5
  print(f't = {t}')
  print()

  N_t, events = ejercicio2a(t)

  print(f'(a) refinamiento')
  res_a = [['evento', 'tiempo']]
  for event in events: res_a.append([event])
  print(tabulate(res_a, headers='firstrow', showindex=range(1, N_t+1)))
  print()

  N_t_imp, events_imp = ejercicio2b(t)

  print(f'(b) refinamiento mejorado')
  res_b = [['evento', 'tiempo']]
  for event in events_imp: res_b.append([event])
  print(tabulate(res_b, headers='firstrow', showindex=range(1, N_t_imp+1)))


def X():
  x = random() * 4 - 2
  y = random() * 2 - 1
  return (x/2)**2 + y**2 <= 1

def ejercicio4b():
  return mean(SIMS, X)

def ejercicio4c():
  z_alpha_2 = 1.96
  L = 0.1

  area, var, sims = rate_interval(z_alpha_2, L, X, scale=8)
  IC = (
    round(area - z_alpha_2 * sqrt(var / sims), 5),
    round(area + z_alpha_2 * sqrt(var / sims), 5)
  )
  return area, IC, sims

def ejercicio4():
  print()
  print('--- Ejercicio 4 ---')
  print()

  p = ejercicio4b()
  res_b = [['(b)', 'sims.', 'proporcion'], ['', SIMS, p]]
  print(tabulate(res_b, headers='firstrow'))
  print()

  area, IC, sims = ejercicio4c()
  res_c = [['(c)', 'sims.', 'area', 'IC(95%)'], ['', sims, area, IC]]
  print(tabulate(res_c, headers='firstrow'))
  print()


def main():
  ejercicio1()
  ejercicio2()
  ejercicio4()

if __name__ == '__main__':
  main()