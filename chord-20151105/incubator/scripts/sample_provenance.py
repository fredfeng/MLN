#!/usr/bin/env python3
# imports {{{
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from functools import partial
from learn_pessimistic import parse_global_provenance, sample_provenance
from random import randrange, seed
from util import posint

import json
import re
import sys

# }}}
# command line parsing {{{
argparser = ArgumentParser(description='''\
  Inputs global provenances; outputs samples.

  NOTE:
    output = provenance -> observation list
    observation = missing * justification list   (one justification per vertex)
    missing = hyperparameter list
    justification = forward_justification * nonforward_justification
    X_justification = hyperparameter list
''', formatter_class=RawDescriptionHelpFormatter)
argparser.add_argument('-big', type=posint,
  default=10,
  help='which value of the parameters is considered big')
argparser.add_argument('-i', type=re.compile,
  default=re.compile(r'(H|O)K$'),
  help='how input tuples look (regexp)')
argparser.add_argument('-o', type=re.compile,
  default=re.compile(r'unsafeDowncast|polySite'),
  help='how output tuples look (regexp for prefix)')
argparser.add_argument('-count', type=posint,
  default=10,
  help='samples per provenance (default 10)')
argparser.add_argument('-independent', action='store_true',
  help="don't track dependencies")
argparser.add_argument('-save-samples',
  default='samples-{}.json'.format(randrange(16 ** 2)),
  help='where to save samples ')
argparser.add_argument('-seed', type=posint,
  default=101,
  help='random seed (default: 101)')
argparser.add_argument('provenances', nargs='+',
  help='from where to read global provenances')

# }}}
# main {{{

def main():
  args = argparser.parse_args()
  seed(args.seed)

  samples = {}
  sample_provenance_ = partial(
      sample_provenance, args.big, args.i, args.o, args.count, args.independent)
  for provenance_filename in args.provenances:
    try:
      with open(provenance_filename, 'r') as provenance_file:
        sys.stderr.write('I: parsing provenance {}\n'.format(provenance_filename))
        provenance = parse_global_provenance(provenance_file)
        sys.stderr.write('I: sampling provenance {}\n'.format(provenance_filename))
        samples[provenance_filename] = sample_provenance_(provenance)
    except Exception as e:
      sys.stderr.write('E: {}\n'.format(e))
      raise
  with open(args.save_samples, 'w') as f:
    json.dump(samples, f, indent=1, sort_keys=True)

if __name__ == '__main__':
  main()

# }}}
