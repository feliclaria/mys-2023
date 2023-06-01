import continuous as cont
from random import random
from time import time
from prettytable import PrettyTable
import math

def ex1():
  n = 0
  E_n = 0
  S_n = 0

  while n < 100 or S_n * 100 >= n:
    E_prev = E_n
    S_prev = S_n
    
    n += 1

    X_n = cont.normal(0, 1)
    E_n = (X_n - E_prev) / (n+1)
    S_n = (n-1) / n * S_prev + (n+1) * (E_prev - E_n)**2 
  
  table = PrettyTable(['Tamaño de la muestra', 'Media muestral', 'Varianza muestral'])
  table.add_row([n, E_n, S_n])
  print(table)


def main():
  ex1()

if __name__ == '__main__':
  main()
