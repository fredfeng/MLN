#!/usr/bin/env python3
# imports {{{
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from learn_pessimistic import \
    likelihood_lowerbound, \
    likelihood_of_independent_samples, \
    likelihood_upperbound, \
    save_expression, \
    simplify_likelihood
from util import tuplify

import json
import re
import sys
# }}}
# command line parsing {{{
argparser = ArgumentParser(description='''\
  Inputs samples; outputs likelihood bounds.

  Likelihood bounds are multivariate polynomials over hyperparameters.
''', formatter_class=RawDescriptionHelpFormatter)
argparser.add_argument('-save-lowerbound',
  help='where to save the lower bound')
argparser.add_argument('-save-upperbound',
  help='where to save the upper bound')
argparser.add_argument('-skip', type=re.compile,
  default=re.compile(r'di2un9i8da023esddowi'),
  help='skip samples from some provenances (regexp)')
argparser.add_argument('samples',
  help='which samples to process')

# }}}
# helpers {{{
def filter_samples(samples, skip):
  sys.stderr.write('I: filtering samples\n')
  sys.stderr.flush()
  samples_as_list = []
  for provenance_name, provenance_samples in samples.items():
    if skip.search(provenance_name) is None:
      samples_as_list.extend(provenance_samples)
  return tuplify(samples_as_list)


def save_likelihood(filename, compute, samples):
  if filename is None:
    return
  sys.stderr.write('I: computing likelihood for {}\n'.format(filename))
  sys.stderr.flush()
  def compute_and_simplify(one_sample):
    return simplify_likelihood(compute(one_sample))
  bound = likelihood_of_independent_samples(compute_and_simplify, samples)
  bound = simplify_likelihood(bound)
  save_expression(bound, filename)

# }}}
# main {{{

def main():
  args = argparser.parse_args()
  with open(args.samples, 'r') as f:
    sys.stderr.write('I: load {}\n'.format(args.samples))
    samples = json.load(f)
  samples = filter_samples(samples, args.skip)
  save_likelihood(args.save_lowerbound, likelihood_lowerbound, samples)
  save_likelihood(args.save_upperbound, likelihood_upperbound, samples)


if __name__ == '__main__':
  main()
# }}}
