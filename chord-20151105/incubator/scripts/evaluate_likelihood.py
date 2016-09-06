#!/usr/bin/env python3
# import {{{
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from functools import partial
from learn_pessimistic import \
  eval_log_expression, \
  likelihood_lowerbound, \
  likelihood_upperbound, \
  load_parameters, \
  tuplify

import json
import sys
# }}}
# command line parsing {{{
argparser = ArgumentParser(description='''\
  Inputs samples and models; outputs log-likelihood bounds.
''', formatter_class=RawDescriptionHelpFormatter)
argparser.add_argument('-v', '--verbose', action='store_true',
  help='print table header')
argparser.add_argument('-samples', required=True,
  help='from where to read samples')
argparser.add_argument('-models', nargs='+', required=True,
  help='from where to load models')

# }}}
# helper {{{
def get_parameter(name, assignment, p):
  if p not in assignment:
    sys.stderr.write('W: {} unknown in {}; using 1\n'.format(p, name))
    assignment[p] = 1
  return assignment[p]


# }}}
# main {{{
def main():
  args = argparser.parse_args()
  models = []
  for m in args.models:
    sys.stderr.write('I: load {}\n'.format(m))
    with open(m, 'r') as f:
      models.append((m, load_parameters(f)))
  with open(args.samples, 'r') as f:
    sys.stderr.write('I: load {}\n'.format(args.samples))
    sys.stderr.flush()
    samples = json.load(f)
  if args.verbose:
    for model_name, _ in models:
      sys.stdout.write(' {}-lb {}-ub'.format(model_name, model_name))
    sys.stdout.write('\n')
  for provenance_name in sorted(samples.keys()):
    sys.stderr.write('I: processing {}\n'.format(provenance_name))
    provenance_samples = samples[provenance_name]
    for one_sample in provenance_samples:
      one_sample = tuplify(one_sample)
      lowerbound = likelihood_lowerbound(one_sample)
      upperbound = likelihood_upperbound(one_sample)
      for model_name, parameters in models:
        get = partial(get_parameter, model_name, parameters)
        low = eval_log_expression(lowerbound, get)
        upp = eval_log_expression(upperbound, get)
        sys.stdout.write(' {} {}'.format(low, upp))
      sys.stdout.write('\n')

if __name__ == '__main__':
  main()
# }}}
