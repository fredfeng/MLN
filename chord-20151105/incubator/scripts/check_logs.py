#!/usr/bin/env python3

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from util import get_logs, get_result

import sys
import util

argparser = ArgumentParser(description='''\
  Checks if outcomes of common logs correspond.
''', formatter_class=RawDescriptionHelpFormatter)

argparser.add_argument('logdirs', nargs='*',
  help='directories with logs to compare')
argparser.add_argument('-v', action='store_true',
  help='verbose')
argparser.add_argument('-oo', type=util.posint,
  default=util.timeout_limit,
  help='timeout')

def report(prefix, results, log):
  sys.stdout.write('{}:'.format(prefix))
  for r in results:
    sys.stdout.write(' {:10}'.format(r))
  sys.stdout.write(' {}\n'.format(log))

def main():
  args = argparser.parse_args()
  util.timeout_limit = args.oo
  sys.stdout.write('dirs:')
  for d in args.logdirs:
    sys.stdout.write(' {}'.format(d))
  sys.stdout.write('\n')
  logs_of_dir = { d : set(get_logs(d)) for d in args.logdirs }
  def wrap_get_result(directory, log):
    if log not in logs_of_dir[directory]:
      return 'none'
    return get_result(Path(directory, log))
  all_logs = sorted(set(l for ls in logs_of_dir.values() for l in ls))
  bad = 0
  for log in all_logs:
    results = [wrap_get_result(d, log) for d in args.logdirs]
    ok_results = [r for r in results if r != 'none' and r != 'limit']
    if len(set(ok_results)) > 1:
      bad += 1
      report('bad', results, log)
    elif args.v:
      report(' ok', results, log)
  sys.stdout.write('summary: total {} bad {}\n'.format(len(all_logs), bad))

if __name__ == '__main__':
  main()
