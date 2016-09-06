#!/usr/bin/env python3

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from glob import glob
from random import randrange, shuffle
from signal import SIGTERM
from shutil import rmtree
from sys import stderr, stdout
from subprocess import Popen, TimeoutExpired
from time import time
from util import posint, timeout_limit

import os
import os.path
import re

argparser = ArgumentParser(description='''\
  Runs several clients on several projects, and saves logs.
''', formatter_class=RawDescriptionHelpFormatter)

argparser.add_argument('-make-script', action='store_true',
  help='make script mode')
argparser.add_argument('-clients',
  default='clients.txt',
  help='clients to run (used only if -make-script)')
argparser.add_argument('-benchmarks',
  default='benchmarks.txt',
  help='benchmarks to run (used only if -make-script)')
argparser.add_argument('-maxq', type=int,
  default=1000000,
  help='queries per (client, benchmark) pair (used only if -make-script)')
argparser.add_argument('-script',
  default='script.txt',
  help='execution script (created if -make-script; used otherwise)')
argparser.add_argument('-model',
  default='model',
  help='probability model (used only in probabilistic mode)')
argparser.add_argument('-timeout', type=posint,
  default=timeout_limit,
  help='timeout in seconds (not strict, though)')
argparser.add_argument('-save-provenances', action='store_true',
  help='save provenances')
argparser.add_argument('-skip', type=int,
  default=0,
  help='skips beginning of the script')
argparser.add_argument('-debug', action='store_true',
  help='enable chord debugging output')

incubator_dir = os.environ['CHORD_INCUBATOR']
bench_dir = os.environ['PJBENCH']
random_suffix = '{:x}-{:02x}'.format(int(time()), randrange(16 ** 2))
temp_outdir = os.path.join\
  ( os.getcwd(), 'chord-tmp-{}'.format(random_suffix) )

chord_settings = \
  { 'chord.experiment.boolDomain' : 'true'
  , 'chord.experiment.model.ruleProbability.scale' : 1000
  , 'chord.out.dir' : temp_outdir }

chord_settings_delta = \
  { 'optimistic' :
    { 'chord.experiment.likelyPossible' : 'true' }
  , 'pessimistic' :
    { 'chord.experiment.likelyPossible' : 'false' }
  , 'probabilistic' :
    { 'chord.experiment.likelyPossible' : 'false'
    , 'chord.experiment.model.class' :
      'chord.analyses.experiment.classifier.RuleProbabilityModel' } }

argparser.add_argument('-mode',
  choices=sorted(chord_settings_delta.keys()),
  default='optimistic',
  help='analysis running mode (not used if -make-script)')

maxsat_solvers = \
  { 'mifumax' : 'chord.analyses.experiment.solver.Mifumax'
  , 'mcsls' : 'chord.analyses.experiment.solver.Mcsls' }

argparser.add_argument('-maxsat', nargs='*',
  choices=sorted(maxsat_solvers.keys()),
  default=['mifumax'],
  help='which solver to use')

# Almost like subprocess.call, but on timout it tries to kill not only the child
# but its descendants as well.
def call(*popenargs, timeout=None, **kwargs):
  with Popen(*popenargs, start_new_session=True, **kwargs) as p:
    try:
      return p.wait(timeout=timeout)
    except:
      os.killpg(p.pid, SIGTERM)
      p.wait()
      raise

def run_chord(benchmark, settings, timeout):
  stdout.write('RUN_CHORD {} {}\n'.format(benchmark, settings))
  cmd = \
    [ os.path.join(incubator_dir, 'runner.pl')
    , '-foreground'
    , '-program={}'.format(benchmark)
    , '-analysis=experiment'
    , '-mode=serial' ]
  for k, v in settings.items():
    cmd += ['-D', '{}={}'.format(k, v)]
  stdout.flush()
  try:
    call(cmd, timeout=timeout+400)
  except TimeoutExpired:
    stdout.write('TIMEOUT {}\n'.format(timeout + 400))

def get_queries(client, benchmark, timeout):
  stdout.write('GET_QUERIES {} {}\n'.format(client, benchmark))
  settings = chord_settings.copy()
  settings.update(
    { 'chord.experiment.onlyReportQueries' : 'true'
    , 'chord.experiment.client' : client })
  run_chord(benchmark, settings, timeout)
  qs = []
  with open(os.path.join(temp_outdir, 'log.txt'), 'r') as f:
    for line in f:
      if line.startswith('MAIN: QUERIES'):
        qs = line.split()[2:]
        break
  stdout.write('GOT_QUERIES {} {}\n'.format(len(qs), ' '.join(qs)))
  rmtree(temp_outdir, ignore_errors=True)
  return qs

def tidy(s):
  return re.sub('[^a-zA-Z0-9]','_',s)

def save_logs(benchmark, client, query, outdir):
  stdout.write('SAVE_LOGS {} {} {} {}\n'.format(benchmark, client, query, outdir))
  log_src = os.path.join(temp_outdir, 'log.txt')
  log_tgt = os.path.join(outdir, tidy('log-{}-{}-{}'.format(benchmark, client, query)))
  try:
    os.rename(log_src, log_tgt)
  except Exception:
    stdout.write('SAVE_LOGS FAILED log.txt\n')
  p_prefix = os.path.join(temp_outdir, 'provenance')
  for p_src in glob('{}*'.format(p_prefix)):
    try:
      p_suffix = p_src[len(p_prefix):]
      p_tgt = os.path.join(
          outdir,
          tidy('provenance-{}-{}-{}-{}'.format(benchmark, client, query, p_suffix)))
      os.rename(p_src, p_tgt)
    except Exception as e:
      stdout.write('SAVE_LOGS FAILED {} {}\n'.format(p, e))


def process(client, benchmark, query, mode, outdir, timeout):
  stdout.write('PROCESS {} {} {} {} {}\n'.format(
    client, benchmark, query, outdir, timeout))
  s = chord_settings.copy()
  s.update(chord_settings_delta[mode])
  s['chord.experiment.client'] = client
  s['chord.experiment.query'] = query
  run_chord(benchmark, s, timeout)
  save_logs(benchmark, client, query, outdir)
  rmtree(temp_outdir, ignore_errors=True)


def make_script(clients_file, benchmarks_file, q_count, script_file, timeout):
  stdout.write('MAKE_SCRIPT {} {} {} {} {}\n'.format(
    clients_file, benchmarks_file, q_count, script_file, timeout))
  with open(clients_file, 'r') as f:
    clients = [x.strip() for x in f.readlines()]
  stdout.write('CLIENTS {}\n'.format(' '.join(clients)))
  with open(benchmarks_file, 'r') as f:
    benchmarks = [x.strip() for x in f.readlines()]
  stdout.write('BENCHMARKS {}\n'.format(' '.join(benchmarks)))
  with open(script_file, 'w') as out:
    for c in clients:
      for b in benchmarks:
        queries = get_queries(c, b, timeout)
        shuffle(queries)
        queries = queries[:q_count]
        stdout.write('KEPT_QUERIES {}\n'.format(' '.join(queries)))
        out.write('{} {} {}\n'.format(c, b, ' '.join(queries)))

def set_model(mode, model_prefix, benchmark):
  if mode != 'probabilistic':
    return
  model = os.path.join(os.getcwd(), '{}-{}'.format(model_prefix, benchmark))
  if not os.path.exists(model):
    model = os.path.join(os.getcwd(), model_prefix)
  if not os.path.exists(model):
    stderr.write('E: cannot find {}\n'.format(model))
    return
  chord_settings_delta['probabilistic']['chord.experiment.model.loadFile'] =\
    model

def get_schedule(script_file, skip):
  script = []
  with open(script_file, 'r') as f:
    for line in f:
      script.append(line.split())
  n = max(len(xs) for xs in script)
  schedule = []
  for i in range(2, n):
    for xs in script:
      if i < len(xs):
        schedule.append((xs[0], xs[1], xs[i]))
  return schedule[skip:]


def run_script(script, skip, timeout, model_prefix, mode, outdir):
  chord_settings['chord.experiment.timeout'] = str(timeout)
  chord_settings['chord.experiment.solver.timeout'] = str(timeout)
  os.mkdir(outdir)
  for client, benchmark, query in get_schedule(script, skip):
    set_model(mode, model_prefix, benchmark)
    try:
      process(client, benchmark, query, mode, outdir, timeout)
    except Exception as e:
      stdout.write('EXCEPTION {} {} {} {}\n'.format(
        client, benchmark, query, e))

def main():
  args = argparser.parse_args()
  chord_settings['chord.experiment.saveGlobalProvenance'] =\
  chord_settings['chord.experiment.accumulate'] =\
    'true' if args.save_provenances else 'false'
  chord_settings['chord.experiment.debug'] =\
  chord_settings['chord.experiment.solver.debug'] =\
    'true' if args.debug else 'false'
  chord_settings['chord.experiment.solvers'] =\
    ':'.join(maxsat_solvers[x] for x in args.maxsat)
  if args.make_script:
    make_script(args.clients, args.benchmarks, args.maxq, args.script, args.timeout)
  else:
    outdir = '{}-logs-{}'.format(args.mode, random_suffix)
    run_script(args.script, args.skip, args.timeout, args.model, args.mode, outdir)

if __name__ == '__main__':
  main()

