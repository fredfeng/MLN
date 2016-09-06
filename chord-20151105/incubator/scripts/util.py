#!/usr/bin/env python3

from pathlib import Path

import re

name_re = re.compile('log_(.*)_(downcast|polysite)_(.*)_$')
iter_line_re = re.compile('MAIN: .* iteration ([0-9]+) +absCost +([0-9])+ +timestamp +([0-9]+)')
timeout_line_re = re.compile('MAIN: TIMEOUT timestamp ([0-9]+)')
oom_line_re = re.compile('OutOfMemoryError')
total_time_re = re.compile('Total time: ([0-9]+):([0-9])+:([0-9])+:([0-9]+) hh:mm:ss:ms')
impossible_re = re.compile('MAIN: impossible queries: 1')
ruledout_re = re.compile('MAIN: ruled out queries: 1')
done_re = re.compile('MAIN: solveUsing loop-end timestamp ([0-9]+)')

timeout_limit = 3600


def tuplify(xs):
  if type(xs) == str:
    return xs
  try:
    return tuple(tuplify(x) for x in xs)
  except TypeError:
    return xs


def unit(s):
  r = float(s)
  if not (0 < r and r <= 1.0):
    raise ValueError
  return r

def posint(s):
  i = int(s)
  if not (i > 0):
    raise ValueError
  return i


def parse_log_name(name):
  m = name_re.match(name)
  if m:
    return (m.group(1), m.group(2), m.group(3))


def get_logs(d):
  rs = []
  for f in Path(d).iterdir():
    if parse_log_name(f.name):
      rs.append(f.name)
  return rs


# Does not include setup time and saving provenance time.  The setup time is
# debatable. It's probably OK to leave out, since the setup is done in exactly
# the same way in the old and new algos. The provenance saving time is not
# debatable. It clearly shouldn't be counted, because it will normaly *not* be
# done by a user.
def get_total_time(log):
  start_time = None
  stop_time = None
  with log.open() as f:
    for line in f:
      m = iter_line_re.match(line)
      if m and int(m.group(1)) == 1:
        assert start_time is None
        start_time = int(m.group(3))
      m = done_re.match(line)
      if m:
        assert stop_time is None
        stop_time = int(m.group(1))
    assert (stop_time is None or start_time is not None)
    if start_time is not None and stop_time is not None:
      r = (1e-9) * (stop_time - start_time)
      if r > timeout_limit:
        return None
      return r
    else:
      return None

def get_iteration_count(log):
  if get_total_time(log) is None:
    return -1
  with log.open() as f:
    n = -1
    for line in f:
      m = iter_line_re.match(line)
      if m:
        n = max(n, int(m.group(1)))
    return n

def get_result(log):
  if get_total_time(log) is None:
    return 'limit'
  impossible = ruled_out = False
  with log.open() as f:
    for line in f:
      impossible |= bool(impossible_re.match(line))
      ruled_out |= bool(ruledout_re.match(line))
  assert (impossible != ruled_out)
  return 'impossible' if impossible else 'ruled_out'
