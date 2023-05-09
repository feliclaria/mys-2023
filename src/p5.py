from random import random
import math

def ex1a():
  U = random()
  if U < 0.25: return 2 + 2 * math.sqrt(U)
  else: return 6 - 6 * math.sqrt((1-U) / 3)

def ex1b():
  U = random()
  if U < 0.6: return math.sqrt(324 + 420 * U) / 6 - 3
  else: return math.cbrt((35 * U - 19) / 2)

def ex1c():
  return None

def main():
  return None

if __name__ == '__main__':
  main()