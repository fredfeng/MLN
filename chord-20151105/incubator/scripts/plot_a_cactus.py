#!/usr/bin/env python3

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from util import get_iteration_count, get_logs, get_total_time

import matplotlib.pyplot as plt
import util

argparser = ArgumentParser(description='''\
  Makes cactus plots, for iteration count and for total runtime.
  Some logs are ignored:
    (a) those not occurring in all input directories
    (b) those with a total time over the given timeout
''', formatter_class=RawDescriptionHelpFormatter)

argparser.add_argument('-i',
  default='iter-cactus.png',
  help='(default: iter-cactus.png)')
argparser.add_argument('-t',
  default='time-cactus.png',
  help='(default: time-cactus.png)')
argparser.add_argument('-l',
  help='labels, comma separated')
argparser.add_argument('-show', action='store_true',
  help="just show, don't save")
argparser.add_argument('-oo', type=util.posint,
  default=util.timeout_limit,
  help='timeout in seconds (default: 900)')
argparser.add_argument('dirs', nargs='*',
  help='directories with logs')


def get_common_logs(dirs):
  common_logs = None
  for d in dirs:
    logs = set(get_logs(d))
    if common_logs is None:
      common_logs = logs
    else:
      common_logs &= logs
  return sorted(common_logs) if common_logs is not None else []


rename = {}
def get_label(d):
  return rename[d] if d in rename else d

show = False
def save_and_reset(outfn):
  if show:
    plt.show()
  else:
    plt.tight_layout()
    with open(outfn, 'w') as out:
      plt.savefig(out, dpi=256)
  plt.clf()


def generate_plot(dirs, logs, get, out, ylabel, ymax=None):
  plt.rc('axes', color_cycle=['r','g','b','y','c','m','k'])
  plt.rc('font', size=5.5)
  plt.figure(figsize=(4,2))
  lw_inc = 0.2
  lw = 1 + lw_inc * len(dirs)
  for d in dirs:
    ys = [get(Path(d, l)) for l in logs]
    ys = [y for y in ys if y is not None]
    ys.append(0)
    ys.sort()
    plt.plot(ys, '-', label=get_label(d), linewidth=lw, alpha=0.9)
    lw -= lw_inc
  plt.xlim(xmin=0)
  plt.ylim(ymin=0)
  if ymax is not None:
    plt.ylim(ymax=ymax)
  plt.ylabel(ylabel)
  plt.xlabel('number of solved queries')
  plt.legend(loc='best')
  save_and_reset(out)


def main():
  global show
  args = argparser.parse_args()
  show = args.show
  util.timeout_limit = args.oo
  if args.l is not None:
    ls = args.l.split(',')
    for i in range(min(len(args.dirs), len(ls))):
      rename[args.dirs[i]] = ls[i]
  logs = get_common_logs(args.dirs)
  def get_iter(log):
    r = get_iteration_count(log)
    return r if r > 0 else None
  generate_plot(args.dirs, logs, get_iter, args.i, 'iterations')
  generate_plot(args.dirs, logs, get_total_time, args.t, 'seconds', ymax=args.oo)


if __name__ == '__main__':
  main()
