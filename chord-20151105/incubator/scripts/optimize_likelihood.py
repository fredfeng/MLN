#!/usr/bin/env python3
# imports {{{
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from learn_pessimistic import \
  apply_substitution, \
  mk_var, \
  optimize_with_coordinate_ascent, \
  optimize_with_gradient_ascent, \
  optimize_with_hopping, \
  optimize_with_scipy, \
  random_parameters, \
  save_parameters, \
  simplify_likelihood
from random import uniform

import json
import sys
import util
# }}}
# command line parsing {{{
argparser = ArgumentParser(description='''\
  Inputs a likelihood; outputs a model.
  That is, it maximizes a multivariate polynomial, on [0,1]^n.
''', formatter_class=RawDescriptionHelpFormatter)
argparser.add_argument('-initial_step', type=util.unit,
  default=0.1,
  help='initial step size (for optimizer hill)')
argparser.add_argument('-a', type=util.unit,
  default=0.99,
  help='step decrease factor (for optimizer hill)')
argparser.add_argument('-iterations', type=util.posint,
  default=10,
  help='number of iterations (for optimizers hill and coord)')
argparser.add_argument('-optimizer',
 choices=['hill', 'coord', 'slsqp', 'hopping'],
 default='coord',
 help='which optimizer to use (default: coord)')
argparser.add_argument('-coarse', action='store_true',
  help='force hyperparameters to be equal')
argparser.add_argument('-model', required=True,
  help='where to write the hyperparameters')
argparser.add_argument('likelihood',
  help='which likelihood to optimize')

# }}}
# main {{{
def main():
  args = argparser.parse_args()
  with open(args.likelihood, 'r') as f:
    sys.stderr.write('I: load likelihood\n')
    sys.stderr.flush()
    likelihood = json.load(f)
  with open(args.model, 'w') as out:
    sys.stderr.write('I: optimize ({})\n'.format(args.optimizer))
    sys.stderr.flush()
    parameters = random_parameters(likelihood, [])
    if args.coarse:
      original_parameters = sorted(parameters.keys())
      substitution = { p : mk_var('theta') for p in original_parameters }
      likelihood = apply_substitution(likelihood, substitution)
      likelihood = simplify_likelihood(likelihood)
      parameters = { 'theta' : uniform(0, 1) }
    if args.optimizer == 'hopping':
      parameters = optimize_with_hopping(likelihood, args.iterations)
    elif args.optimizer == 'slsqp':
      parameters = optimize_with_scipy(likelihood, args.iterations)
    elif args.optimizer == 'hill':
      parameters = optimize_with_gradient_ascent(
          likelihood, args.iterations, args.initial_step, args.a, parameters)
    elif args.optimizer == 'coord':
      parameters = optimize_with_coordinate_ascent(
          likelihood, args.iterations, parameters)
    else:
      assert False
    if args.coarse:
      theta = parameters['theta']
      parameters = { p : theta for p in original_parameters }
    save_parameters(out, parameters)


if __name__ == '__main__':
  main()
# }}}
