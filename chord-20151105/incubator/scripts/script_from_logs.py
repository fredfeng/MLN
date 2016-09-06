#!/usr/bin/env python3

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import defaultdict
from pathlib import Path

import re
import util

argparser = ArgumentParser(description='''\
  Inputs logs; outputs script that mentions only easy queries.
''', formatter_class=RawDescriptionHelpFormatter)

argparser.add_argument('logdir',
  help='directory with log files')
argparser.add_argument('-script',
  default='easy-script.txt',
  help='where to save the output')
argparser.add_argument('-oo', type=util.posint,
  default=util.timeout_limit,
  help='timeout')

def queryname(q):
  q = re.sub('_','(',q,count=1)
  q = re.sub('_',',',q)
  return q + ')'

def main():
  args = argparser.parse_args()
  util.timeout_limit = args.oo
  easy_queries = defaultdict(list)
  for log in util.get_logs(args.logdir):
    (benchmark, client, query) = util.parse_log_name(log)
    if util.get_total_time(Path(args.logdir, log)) is not None:
      easy_queries[(client, benchmark)].append(queryname(query))
  with open(args.script, 'w') as out:
    for (client, benchmark), qs in easy_queries.items():
      out.write('{} {} {}\n'.format(client, benchmark, ' '.join(qs)))

if __name__ == '__main__':
  main()

