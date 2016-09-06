#!/usr/bin/env python3

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import defaultdict, deque

import sys

argparser = ArgumentParser(description='''
  Assuming pessimistic mode, prints the derivation found by the optimizer.
''', formatter_class=RawDescriptionHelpFormatter)

argparser.add_argument('model', help='mifumax...out')
argparser.add_argument('derivation', help='refine...explicit')
argparser.add_argument('tuplemap')
argparser.add_argument('-reach', help='slice the derivation wrt this tuple')


def parse_model(model_file):
  with open(model_file) as f:
    for line in f:
      if line.startswith('v '):
        model = set(int(x) for x in line[2:].split() if int(x) > 0)
        model.add(0) # represents 'true' (empty rhs)
  return model


def parse_tuplemap(tuplemap_file):
  with open(tuplemap_file) as f:
    idx = dict()
    for line in f:
      ws = line.split()
      idx[ws[1]] = int(ws[0])
    idx[''] = 0 # empty rhs
  return idx


def parse_derivation(derivation_file):
  derivation = []
  with open(derivation_file) as f:
    for line in f:
      line = line.split(':')[1].strip().split('<-')
      left, right = line[0].strip(), line[1].strip()
      left = [x.strip() for x in left.split('|')] if left != '' else []
      derivation.append((left, right))
  return derivation


def filter_by_model(old_derivation, model, index):
  def is_set(t):
    i = int(t[3:]) if t.startswith('Aux') else index[t]
    return i in model
  new_derivation = []
  for xs, y in old_derivation:
    if is_set(y):
      xs = [x for x in xs if is_set(x)]
      if xs != []:
        new_derivation.append((xs, y))
  return new_derivation


# Does not select one derivation of target; it selects *all*.
# Also, puts constraints in BFS order.
def filter_by_target(old_derivation, target):
  if target is None:
    return old_derivation

  # Prepare graph.
  by_head = defaultdict(list)
  for xs, y in old_derivation:
    by_head[y].append(xs)

  # Do BFS, and record new derivation.
  q = deque([target])
  seen = set(q)
  new_derivation = []
  while q:
    y = q.popleft()
    for xs in by_head[y]:
      for x in xs:
        if x not in seen:
          q.append(x)
          seen.add(x)
      new_derivation.append((xs, y))
  return new_derivation


def print_derivation(derivation):
  def print_constraint(xs, y):
    def f(xs):
      return '|'.join(xs) if xs != [] else 'False'
    def g(y):
      return y if y != '' else 'True'
    sys.stdout.write('{} <- {}\n'.format(f(xs), g(y)))
  for xs, y in derivation:
    print_constraint(xs, y)


def main():
  args = argparser.parse_args()
  model = parse_model(args.model)
  index = parse_tuplemap(args.tuplemap)
  derivation = parse_derivation(args.derivation)
  derivation = filter_by_model(derivation, model, index)
  derivation = filter_by_target(derivation, args.reach)
  print_derivation(derivation)


if __name__ == '__main__':
  main()
