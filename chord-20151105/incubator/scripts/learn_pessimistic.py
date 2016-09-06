#!/usr/bin/env python3
# imports {{{

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import defaultdict, deque
from functools import partial
from itertools import chain
from math import ceil, exp, log, log1p
from random import choice, randrange, sample, seed, shuffle, uniform
from re import compile, escape
from scipy.optimize import basinhopping, minimize
from time import process_time
from util import tuplify

import json
import numpy as np
import sys

# }}}
# constants {{{
SAFE = True   # DBG
infinity = float('inf')

# }}}
# helpers for handling provenances {{{
def check_provenance(provenance):
  if not SAFE:
    return
  ok = True
  types = { n : args for n, args in provenance['types'] }
  def_vs = set(provenance['vertices'])
  def_rs = set(provenance['rules'])
  def_cs = set(x[0] for x in provenance['contexts'])
  if len(def_vs) != len(provenance['vertices']):
    sys.stderr.write('E: multiply defined vertices\n')
    ok = False
  if len(def_rs) != len(provenance['rules']):
    sys.stderr.write('E: multiply defined rules\n')
    ok = False
  if len(types) != len(provenance['types']):
    sys.stderr.write('E: multiply defined types\n')
    ok = False
  used_vs = set(x for _, x, _ in provenance['arcs'])
  used_vs |= set(y for _, _, ys in provenance['arcs'] for y in ys)
  used_rs = set(r for r, _, _ in provenance['arcs'])
  used_cs = set(x[-1] for x in provenance['contexts'])
  used_cs |= set(x[1][i] for x in def_vs for i in range(len(x[1]))
      if x[0] in types and types[x[0]][i] == 'DomC')
  undef_vs = used_vs - def_vs
  undef_rs = used_rs - def_rs
  undef_cs = used_cs - def_cs
  undef_types = set(x[0] for x in used_vs
      if x[0] not in types)
  bad_arity = set(x for x in used_vs
      if x[0] in types and len(x[1]) != len(types[x[0]]))
  if undef_vs:
    sys.stderr.write('E: undefined vertices: {}\n'.format(undef_vs))
    ok = False
  if undef_rs:
    sys.stderr.write('E: undefined rules: {}\n'.format(undef_rs))
    ok = False
  if undef_cs:
    sys.stderr.write('E: undefined contexts: {}\n'.format(undef_cs))
    ok = False
  if undef_types:
    sys.stderr.write('E: undefined types: {}\n'.format(sorted(list(undef_types))))
    ok = False
  if bad_arity:
    sys.stderr.write('E: mismatched arity: {}\n'.format(sorted(list(bad_arity))))
    ok = False
  for _, _, ys in provenance['arcs']:
    assert type(ys) == tuple
  assert ok # I want to see the stack trace when this fails


def parse_global_provenance(file):
  def tup(t):
    return (t[0], tuple(t[1]))
  provenance = json.load(file)
  provenance['arcs'] = [(r, tup(x), tuple(tup(y) for y in ys))
      for r, x, ys in provenance['arcs']]
  provenance['vertices'] = [tup(x) for x in provenance['vertices']]
  check_provenance(provenance)
  return provenance


def debug_provenance_size(provenance):
  r = {}
  for k, v in provenance.items():
    r[k] = len(v)
  return r


# Returns a list of lists. The inner lists look like
#   [(0, ('H', (100, 0))), (10, ('H', (100, 10)))]
# (This one corresponds to the parameter H100.)
def find_inputs(provenance, in_re):
  inputs = defaultdict(set)
  for _, x, ys in provenance['arcs']:
    if in_re.match(x[0]):
      sys.stderr.write('W: parameter {} appears in the head of a rule\n'.format(x))
    for y in ys:
      m = in_re.match(y[0])
      if m:
        p = (y[0], y[1][0])
        inputs[p].add((y[1][1], y))
  return [sorted(xs) for xs in inputs.values()]


# Returns a list of tuples that match out_re and also appear in the head of
# some arc of provenance.
def find_outputs(provenance, out_re):
  outputs = set()
  for _, x, _ in provenance['arcs']:
    m = out_re.match(x[0])
    if m:
      outputs.add(x)
  return sorted(outputs)
# }}}
# reachability in provenances {{{
# Keeps those arcs that are reachable from a given set of inputs.
# If only_forward, then keeps only arcs (x <- ys) such that d(x) > max(d(y)),
# where d denotes distance from inputs.
def keep_relevant(provenance, inputs, only_forward=False):

  # forward reachability
  watch = { y : [] for y in provenance['vertices'] }
  this_level, next_level = None, list(inputs)
  done, seen = set(), set(inputs)
  new_arcs = defaultdict(list)
  for arc in provenance['arcs']:
    r, x, ys = arc
    if len(ys) == 0:
      if x not in seen:
        seen.add(x)
        next_level.append(x)
      new_arcs[x].append((r, ys))
    else:
      watch[ys[0]].append(arc)
  while next_level:
    this_level, next_level = next_level, []
    done.update(this_level)
    assert not SAFE or done == seen
    for y in this_level:
      for arc in watch[y]:
        r, x, ys = arc
        if x in done and only_forward:
          continue # not a forward arc
        zs = [z for z in ys if z not in seen]
        if zs == []:
          new_arcs[x].append((r, ys))
          if x not in seen:
            seen.add(x)
            next_level.append(x)
        else:
          watch[zs[0]].append(arc)
    watch[y] = None # don't use later (and save memory)

  # make new provenance
  result = {}
  result['arcs'] = \
    [(r, x, ys) for x, rys in new_arcs.items() for r, ys in rys]
  result['rules'] = sorted(set(r for r, _, _ in result['arcs']))
  result['vertices'] = sorted(seen)
  used_names = set(x[0] for x in seen)
  result['types'] = [t for t in provenance['types'] if t[0] in used_names]
  result['contexts'] = provenance['contexts']

  check_provenance(result)
  return result

# }}}
# sample provenances {{{

def remove_cycling_arcs(provenance):
  provenance['arcs'] = [(r, x, ys) for r, x, ys in provenance['arcs']
      if x not in ys]
  check_provenance(provenance)

# From the global provenance it produces
#   (cheap_provenance, precise_provenance, projection)
# The inputs and outputs of global_provenance are identified using in_re and
# out_re. Only those inputs that have been tried with values both <big and
# >=big are kept in the local provenances: the cheap_provenance keeps the
# smallest value; the precise_provenance keeps the largest value. Local
# provenances, unlike global provenaces, have two extra fields: 'inputs' and
# 'outputs'.
#
# The projection is defined in terms of a relation P between vertices x in the
# cheap provenance and vertices y in the precise provenance. We have xPy when
#   - both x and y are on some path (in their respective provenances) from
#     inputs to outputs; the inputs are defined here as abstraction tuples;
#     the outputs are defined here as queries OR "Deny" tuples
#   - the names of x and y come from a given relation
#     (this relation is the identity most of the time; see name_map below)
#   - x and y have the same types, which implies the same number of arguments
#   - for each pair (x[i], y[i]) of corresponding arguments,
#     - if the domain is DomK, then x[i] <= y[i]
#     - if the domain is DomC, then y[i] projects on x[i] according to the
#       (explicit) projection relation on contexts, in one or more steps
#     - otherwise, x[i] == y[i]
# The projection is P, represented as a mapping from vertices in the precise
# provenance to lists of vertices in the cheap provenance. The projection
# computed as described above is often a function, but this isn't guaranteed.
# NOTE: It also removes cycling arcs
def locals_of_global(global_provenance, big, in_re, out_re):
  types = dict(global_provenance['types'])
  context_projection = { c[0] : c[1] for c in global_provenance['contexts']
      if len(c) == 2 }
  def get_local_inputs():
    inputs = [ts for ts in find_inputs(global_provenance, in_re)
        if ts[0][0] < big and ts[-1][0] >= big]
    return \
        { 'cheap' : [ts[0][1] for ts in inputs]
        , 'precise' : [ts[-1][1] for ts in inputs] }
  def get_slice(inputs):
    local = keep_relevant(global_provenance, inputs)
    local['inputs'] = [x[0][1] for x in find_inputs(local, in_re)]
    local['outputs'] = list(find_outputs(local, out_re))
    assert not SAFE or set(local['inputs']) <= set(inputs)
    return local
  def get_projection(cheap_vertices, precise_vertices):
    def bucket_vertices(xs):
      def b(x):
        def pa(a, t):
          if t == 'DomC':
            return 'C'
          elif t == 'DomK':
            return 'K'
          else:
            return a
        return (x[0], tuple(pa(x[1][i], types[x[0]][i]) for i in range(len(x[1]))))
      buckets = defaultdict(list)
      for x in xs:
        buckets[b(x)].append(x)
      return buckets
    name_map = { x[0] : [x[0]] for x in precise_vertices }
    name_map['COC'] = ['COC', 'COC_1', 'COC_2']
    name_map['COC_1'] = ['COC_1']
    name_map['COC_2'] = []
    def projects(precise, cheap):
      assert cheap[0] in name_map[precise[0]]
      assert types[precise[0]] == types[cheap[0]]
      n = len(cheap[1])
      assert n == len(precise[1]) and n == len(types[cheap[0]])
      for i in range(n):
        if types[cheap[0]][i] == 'DomC':
          context = precise[1][i]
          while context != cheap[1][i] and context in context_projection:
            context = context_projection[context]
          if context != cheap[1][i]:
            return False
        elif types[cheap[0]][i] == 'DomK':
          if cheap[1][i] > precise[1][i]:
            return False
        else:
          if cheap[1][i] != precise[1][i]:
            return False
      return True
    cheap_buckets = bucket_vertices(cheap_vertices)
    precise_buckets = bucket_vertices(precise_vertices)
    projection = defaultdict(list)
    for pb, xs in precise_buckets.items():
      for cheap_name in name_map[pb[0]]:
        cb = tuple((pb[i] if i != 0 else cheap_name) for i in range(len(pb)))
        for y in cheap_buckets[cb]:
          for x in xs:
            if projects(x, y):
              projection[x].append(y)
    return projection
  local_inputs = get_local_inputs()
  cheap_provenance = get_slice(local_inputs['cheap'])
  precise_provenance = get_slice(local_inputs['precise'])
  check_provenance(cheap_provenance)
  check_provenance(precise_provenance)
  projection = get_projection(
      cheap_provenance['vertices'], precise_provenance['vertices'])

  remove_cycling_arcs(cheap_provenance)
  remove_cycling_arcs(precise_provenance)
  return \
      { 'cheap_provenance' : cheap_provenance
      , 'precise_provenance' : precise_provenance
      , 'projection' : projection }


# Returns a list of independent observations. Here is an example observation:
#   (['a', 'b', 'c'], [[...], [(['d', 'e'], ['f']), (['e'], ['f', 'd'])],  ...])
# Here ['a', 'b', 'c'] is the negative observation: We noticed that these arcs
# must be missing from the predictive provenance. Next comes the positive
# observation, which is a list of justifications for each vertex. For example,
# one of the vertices in this example has the following list of justifications
#   [(['d', 'e'], ['f']), (['e'], ['f', 'd'])]
# The two justifications are for two different abstractions. For example,
#   (['d', 'e'], ['f'])
# means that for one abstraction the current vertex needed to be justified, and
# was justified by the forward arcs ['d', 'e'] and the nonforward arcs ['f'].
#
# The 'a', 'b', 'c', ... from above are arcs.
#
# Samples are computed as follows:
#   1. Obtain pairs (R1, T1), ..., (Rn, Tn)
#   2. Compute missing arcs, and remove them from the cheap provenance.
#   3. Compute forward and nonforwards arcs.
# For details of each step, see inline comments.
def sample_provenance(big, in_re, out_re, samples_count, independent, provenance):
  L = locals_of_global(provenance, big, in_re, out_re)
  big, in_re, out_re, provenance = None, None, None, None # don't use later

  result = []
  missing_arcs = set()
  justifications = defaultdict(list)

  def sample_precise_inputs():
    xs = L['precise_provenance']['inputs']
    m = 1 + randrange(len(xs))
    return sample(xs, m)

  def project(ys):
    return frozenset(x for y in ys for x in L['projection'][y])

  def reach_and_project(p1s):
    precise_slice = keep_relevant(L['precise_provenance'], p1s)
    return project(precise_slice['vertices'])

  def get_new_missing_arcs(reachable):
    new_missing_arcs = []
    for arc in L['cheap_provenance']['arcs']:
      _, x, ys = arc
      assert type(reachable) == frozenset
      if x not in reachable and all(y in reachable for y in ys):
        new_missing_arcs.append(arc)
    return frozenset(new_missing_arcs)

  def get_new_justifications(inputs, reachable):
    assert type(inputs) == frozenset
    assert type(reachable) == frozenset
    # Do NOT use defaultdict on the next line!
    new_justifications = { x : ([], []) for x in reachable - inputs }
    cheap_slice = keep_relevant(L['cheap_provenance'], inputs)
    forward_slice = keep_relevant(cheap_slice, inputs, only_forward=True)
    forward_arcs = frozenset(forward_slice['arcs'])
    for arc in cheap_slice['arcs']:
      _, x, ys = arc
      assert x not in inputs
      if x in reachable and all(y in reachable for y in ys):
        forward, nonforward = new_justifications[x]
        if arc in forward_arcs:
          forward.append(arc)
        else:
          nonforward.append(arc)
    return new_justifications

  def contradiction(new_missing_arcs, new_justifications):
    assert type(missing_arcs) == set
    assert type(new_missing_arcs) == frozenset
    def bad(ys):
      return all(y in missing_arcs or y in new_missing_arcs for y in ys)
    for yss in justifications.values():
      for ys, _ in yss:
        if bad(ys):
          return True
    for ys, _ in new_justifications.values():
      if bad(ys):
        return True
    return False

  def record_observation():
    if len(missing_arcs) == 0 and len(justifications) == 0:
      sys.stderr.write('W: predictibility fails for one abstraction\n')
      sys.stderr.flush()
    else:
      result.append((sorted(missing_arcs), sorted(justifications.values())))
    missing_arcs.clear()
    justifications.clear()

  def update_missing_arcs(new_missing_arcs):
    missing_arcs.update(new_missing_arcs)

  def update_justifications(new_justifications):
    for x, j in new_justifications.items():
      justifications[x].append(j)

  last_reported_time = process_time()
  def report_progress(samples_count):
    nonlocal last_reported_time
    now = process_time()
    if now > last_reported_time + 10:
      sys.stderr.write('I: samples still to do: {}\n'.format(samples_count))
      sys.stderr.flush()
      last_reported_time = now

  # This is the main loop.
  while samples_count > 0:
    report_progress(samples_count)
    precise_inputs = sample_precise_inputs()
    cheap_inputs = project(precise_inputs)
    reachable = reach_and_project(precise_inputs)
    new_missing_arcs = get_new_missing_arcs(reachable)
    new_justifications = get_new_justifications(cheap_inputs, reachable)
    if contradiction(new_missing_arcs, new_justifications):
      record_observation()
    else:
      update_missing_arcs(new_missing_arcs)
      update_justifications(new_justifications)
      samples_count -= 1
      if independent:
        record_observation()
  record_observation()

  return result

# }}}
# helpers for handling likelihoods and polynomials {{{

def guarded_log(x):
  eps = 1e-6
  assert -eps <= x and x <= 1 + eps
  if x >= 1:
    return 0
  if x <= 0:
    return float('-inf')
  return log(x)
def guarded_log1p(x):
  eps = 1e-6
  assert -1-eps <= x and x <= 0 + eps
  if x >= 0:
    return 0
  if x <= -1:
    return float('-inf')
  return log1p(x)


# experimental {{{
# I'm trying a more flexible representation for expressions.
#   T = ('add', [T, ..., T])
#   T = ('mul', [T, ..., T])
#   T = ('pow', (T, 123))             x^123
#   T = ('rep', (T, 1023))        1023x
#   T = ('not', T)                  1-x
#   T = ('var', 'foo')
#   T = ('num', 0.76)

def mk_add(es):
  return ('add', tuple(es))
def mk_mul(es):
  return ('mul', tuple(es))
def mk_pow(e, n):
  assert type(n) == int
  return ('pow', (e, n))
def mk_rep(e, n):
  assert type(n) == int
  return ('rep', (e, n))
def mk_not(e):
  return ('not', e)
def mk_var(x):
  assert type(x) == str
  return ('var', x)
def mk_num(v):
  return ('num', v)


def save_expression(E, outfilename):
  with open(outfilename, 'w') as f:
    json.dump(E, f, indent=1)


def load_expression(infilename):
  with open(infilename, 'r') as f:
    return tuplify(json.load(f))


# TODO: use a Visitor?


def partial_eval_expression(E, assignment):
  def do_add(rec, es):
    vs = []
    fs = []
    for e in es:
      f = rec(e)
      if f[0] == 'num':
        vs.append(f[1])
      else:
        fs.append(f)
    if vs:
      fs.append(mk_num(sum(vs)))
    return mk_add(fs)
  def do_mul(rec, es):
    vs = []
    fs = []
    for e in es:
      f = rec(e)
      if f[0] == 'num':
        vs.append(f[1])
      else:
        fs.append(f)
    if vs:
      fs.append(mk_num(exp(sum(guarded_log(v) for v in vs))))
    return mk_mul(fs)
  def do_pow(rec, en):
    e, n = en
    f = rec(e)
    if f[0] == 'num':
      return mk_num(exp(n * guarded_log(f[1])))
    else:
      return mk_pow(f, n)
  def do_rep(rec, en):
    e, n = en
    f = rec(e)
    if f[0] == 'num':
      return mk_num(f[1] * n)
    else:
      return mk_rep(f, n)
  def do_not(rec, e):
    f = rec(e)
    if f[0] == 'num':
      return mk_num(1 - f[1])
    else:
      return mk_not(f)
  def do_var(_, x):
    v = assignment(x)
    if v:
      return mk_num(v)
    else:
      return mk_var(x)
  def do_num(_, v):
    return mk_num(v)
  dispatch  = \
    { 'add' : do_add
    , 'mul' : do_mul
    , 'rep' : do_rep
    , 'pow' : do_pow
    , 'not' : do_not
    , 'var' : do_var
    , 'num' : do_num }
  def go(E):
    op, xs = E
    return dispatch[op](go, xs)
  return go(E)


def eval_log_expression(E, assignment):
  def do_add(rec, es):
    return guarded_log(sum(exp(rec(e)) for e in es))
  def do_mul(rec, es):
    return sum(rec(e) for e in es)
  def do_pow(rec, en):
    e, n = en
    return n * rec(e)
  def do_rep(rec, en):
    e, n = en
    return log(n) + rec(e)
  def do_not(rec, e):
    return guarded_log1p(-exp(rec(e)))
  def do_var(_, x):
    return guarded_log(assignment(x))
  def do_num(_, v):
    return guarded_log(v)
  dispatch  = \
    { 'add' : do_add
    , 'mul' : do_mul
    , 'rep' : do_rep
    , 'pow' : do_pow
    , 'not' : do_not
    , 'var' : do_var
    , 'num' : do_num }
  def eval_log(E):
    op, xs = E
    return dispatch[op](eval_log, xs)
  return eval_log(E)


def eval_expression(E, assignment):
  return exp(eval_log_expression(E, assignment))


# computes d/dx log(E)
def eval_dx_log_expression(E, assignment, X):
  def do_add(rec, es):
    a = b = 0
    for e in es:
      V = eval_expression(e, assignment)
      dV = rec(e)
      a += V * dV
      b += V
    if b == 0:
      return infinity if a > 0 else -infinity
    return a / b
  def do_mul(rec, es):
    return sum(rec(e) for e in es)
  def do_pow(rec, en):
    e, n = en
    assert type(n) == int
    return n * rec(e)
  def do_rep(rec, en):
    return rec(en[0])
  def do_not(rec, e):
    V = eval_expression(e, assignment)
    r = rec(e)
    assert V != 1
    if V == 1:
      return infinity if r > 0 else -infinity
    return V / (V - 1) * r
  def do_var(_, x):
    assert type(x) == str
    return 1 / assignment(X) if x == X else 0
  def do_num(_rec, _v):
    return 0
  dispatch = \
    { 'add' : do_add
    , 'mul' : do_mul
    , 'rep' : do_rep
    , 'pow' : do_pow
    , 'not' : do_not
    , 'var' : do_var
    , 'num' : do_num }
  def go(E):
    op, xs = E
    return dispatch[op](go, xs)
  return go(E)

def vars_of_expression(E):
  result = set()
  def go(E):
    def rec_list(es):
      for e in es:
        go(e)
    def rec_pair(en):
      e, n = en
      assert type(n) == int
      go(e)
    do_add = do_mul = rec_list
    do_pow = do_rep = rec_pair
    do_not = go
    def do_var(x):
      assert type(x) == str
      result.add(x)
    def do_num(_):
      pass
    dispatch  = \
      { 'add' : do_add
      , 'mul' : do_mul
      , 'rep' : do_rep
      , 'pow' : do_pow
      , 'not' : do_not
      , 'var' : do_var
      , 'num' : do_num }
    op, xs = E
    return dispatch[op](xs)
  go(E)
  return result

# Compresses AC operators.
def collect_expression(E):
  def go_ac(rec, es, op1, op2, mk_op1, mk_op2):
    counts = defaultdict(int)
    es = [rec(e) for e in es]
    def C(e):
      op, es = e
      if op == op2:
        counts[es[0]] += es[1]
      elif op == op1:
        for e in es:
          C(e)
      else:
        counts[e] += 1
    for e in es:
      C(e)
    es = []
    for e, c in counts.items():
      if c == 0:
        continue
      elif c == 1:
        es.append(e)
      else:
        es.append(mk_op2(e, c))
    return mk_op1(es)
  def do_add(rec, es):
    return go_ac(rec, es, 'add', 'rep', mk_add, mk_rep)
  def do_mul(rec, es):
    return go_ac(rec, es, 'mul', 'pow', mk_mul, mk_pow)
  def do_pow(rec, en):
    return mk_pow(rec(en[0]), en[1])
  def do_rep(rec, en):
    return mk_rep(rec(en[0]), en[1])
  def do_not(rec, e):
    return mk_not(rec(e))
  def do_var(_, x):
    return mk_var(x)
  def do_num(_, n):
    return mk_num(n)
  dispatch  = \
    { 'add' : do_add
    , 'mul' : do_mul
    , 'rep' : do_rep
    , 'pow' : do_pow
    , 'not' : do_not
    , 'var' : do_var
    , 'num' : do_num }
  def go(E):
    op, es = E
    return dispatch[op](go, es)
  return go(E)


def simplify_likelihood(likelihood):
  # TODO: improve
  return collect_expression(likelihood)


def apply_substitution(E, substitution):
  def do_add(rec, es):
    return mk_add(rec(e) for e in es)
  def do_mul(rec, es):
    return mk_mul(rec(e) for e in es)
  def do_pow(rec, en):
    return mk_pow(rec(en[0]), en[1])
  def do_rep(rec, en):
    return mk_rep(rec(en[0]), en[1])
  def do_not(rec, e):
    return mk_not(rec(e))
  def do_var(_, x):
    return substitution[x] if x in substitution else mk_var(x)
  def do_num(_, v):
    return mk_num(v)
  dispatch = \
    { 'add' : do_add
    , 'mul' : do_mul
    , 'rep' : do_rep
    , 'pow' : do_pow
    , 'not' : do_not
    , 'var' : do_var
    , 'num' : do_num }
  def go(E):
    op, xs = E
    return dispatch[op](go, xs)
  return go(E)


def relpoly_of_cnf(cnf):
  # Give indices 0, 1, 2, ... to the arcs/variables in the CNF.
  index_of_arc, arc_of_index = {}, []
  index_of_theta, theta_of_index = {}, []
  for clause in cnf:
    for arc in clause:
      if arc not in index_of_arc:
        index_of_arc[arc] = len(arc_of_index)
        arc_of_index.append(arc)
      theta, _, _ = arc
      if theta not in index_of_theta:
        index_of_theta[theta] = len(theta_of_index)
        theta_of_index.append(theta)

  m = sum(len(clause) for clause in cnf)
  n = len(arc_of_index)
  time_estimate = m * 2 ** n
  start_time = None
  if time_estimate > 2 ** 20:
    sys.stderr.write('W: relpoly_of_cnf: running algo with ~{} steps\n'.format(
      time_estimate))
    sys.stderr.flush()
    start_time = process_time()

  # Go thru all subsets of variables/arcs.
  terms = []
  for mask in range(1 << n):
    def has_arc(x):
      return ((mask >> index_of_arc[x]) & 1) != 0
    if not all(any(has_arc(x) for x in clause) for clause in cnf):
      continue
    factors = []
    for i in range(n):
      theta, _, _ = arc_of_index[i]
      one_factor = mk_var(theta)
      if ((mask >> i) & 1) == 0:
        one_factor = mk_not(one_factor)
      factors.append(one_factor)
    terms.append(mk_mul(factors))
  relpoly = mk_add(terms)

  if start_time is not None:
    elapsed = process_time() - start_time
    sys.stderr.write('W: ... ~{} steps done in {:.2f} seconds\n'.format(
      time_estimate, elapsed))
    sys.stderr.flush()

  return relpoly


def likelihood_lowerbound(sample):
  def shrink(xss, n):
    # Given a monotone CNF xss, truncate each of its clauses to length at
    # most n.  An attempt is made to keep frequent variables, heuristically.
    count = defaultdict(int)
    for xs in xss:
      for x in xs:
        count[x] += 1
    yss = []
    for xs in xss:
      yss.append(tuple(sorted(xs, key=count.get, reverse=True)[:n]))
    return yss
  def vertex(j):
    forward_justification = tuple(a for a, _ in j)
    approx = shrink(forward_justification, 8)
    return relpoly_of_cnf(approx)
  missing, justification = sample
  factors = []
  factors.extend(mk_not(mk_var(theta)) for theta, _, _ in missing)
  factors.extend(vertex(j) for j in justification)
  return mk_mul(factors)

def likelihood_upperbound(sample):
  def vertex(j):
    # keep one of the longest clauses
    justification = None
    for a, b in j:
      if justification is None or len(justification) < len(a) + len(b):
        justification = a + b
    return mk_not(mk_mul(mk_not(mk_var(theta)) for theta, _, _ in justification))
  missing, justification = sample
  factors = []
  factors.extend(mk_not(mk_var(theta)) for theta, _, _ in missing)
  factors.extend(vertex(j) for j in justification)
  return mk_mul(factors)

def likelihood_of_independent_samples(get_likelihood, samples):
  return mk_mul(get_likelihood(x) for x in samples)


# }}}
# numerical optimization of likelihood {{{
def random_parameters(likelihood, parameters):
  result = dict(parameters)
  for p in vars_of_expression(likelihood):
    if p not in result:
      result[p] = uniform(0, 1)
  return result


# d/dx1 log L, ..., d/dxn log L
def compute_gradient(likelihood, parameters):
  gradient = { p : eval_dx_log_expression(likelihood, parameters.get, p)
      for p in parameters.keys() }
  if False: # DBG
    print('gradient', sorted(gradient.items()))
  return gradient


def update_parameters(parameters, gradient, alpha):
  eps = 0.0001
  max_delta = 0.1
  min_delta = 0
    # with 0 gets stuck in local optima quite often, but converges
    # with eps it often finds a better local optimum, but sometimes doesn't converge
  def snap(x):
    return min(1-eps, max(eps, x))
  max_d = 0
  for p, v in parameters.items():
    d = alpha * gradient[p]
    if v < 2 * eps and v + d < 2 * eps:
      continue
    if v > 1 - 2 * eps and v + d > 1 - 2 * eps:
      continue
    max_d = max(max_d, abs(d))
  if False: # DBG
    print('max_d',max_d)
    print('alpha',alpha)
  if max_d != 0:
    if max_d < min_delta:
      alpha *= min_delta / max_d
    if max_d > max_delta:
      alpha *= max_delta / max_d
  if False: # DBG
    print('adjusted_alpha',alpha)
  return { p : snap(v + alpha * gradient[p]) for p, v in parameters.items() }


def scipy_likelihood(likelihood, parameters, x):
  n = len(parameters)
  if any(not (0 < t and t < 1) for t in x):
    return infinity
  ps = { parameters[i] : x[i] for i in range(n) }
  return -eval_log_expression(likelihood, ps.get)


def scipy_likelihood_jac(likelihood, parameters, x):
  def d(t):
    if not (0 < t):
      print('MIN')
      return -1
    elif not (t < 1):
      print('MAX')
      return 1
    else:
      return 0
  jac = [d(t) for t in x]
  if any(jac):
    return jac
  n = len(parameters)
  ps = { parameters[i] : x[i] for i in range(n) }
  jac = compute_gradient(likelihood, ps)
  return np.array([-jac[parameters[i]] for i in range(n)])


def optimize_with_scipy(likelihood, iterations):
  parameters = list(vars_of_expression(likelihood))
  n = len(parameters)
  epsilon = 1e-9
  res = minimize(
      partial(scipy_likelihood, likelihood, parameters),
      jac=partial(scipy_likelihood_jac, likelihood, parameters),
      x0=[uniform(0, 1) for _ in parameters],
      bounds=[(epsilon,1-epsilon)]*n,
      method='SLSQP',
      options={'disp':True, 'maxiter':iterations})
  return { parameters[i] : res.x[i] for i in range(n) }


# TODO: add derivative
def optimize_with_hopping(likelihood, iterations):
  parameters = list(vars_of_expression(likelihood))
  def mk_fa(xs): # scipy fails with weird error without this wrapping
    return np.array([float(x) for x in xs])
  def objective_f(x):
    return scipy_likelihood(likelihood, parameters, x)
  def accept(**kwargs):
    return all((0 < x and x < 1) for x in kwargs['x_new'])
  res = basinhopping(
      objective_f,
      x0=[uniform(0, 1) for _ in parameters],
      #minimizer_kwargs={'method':'L-BFGS-B', 'jac':False},
      minimizer_kwargs={'method':'SLSQP', 'jac':False},
      accept_test=accept,
      niter=iterations)
  if False: # DBG
    print('OPTIMIZATION RESULT', res)
  return { parameters[i] : res.x[i] for i in range(len(parameters)) }


def estimate_likelihood_quality(likelihood, parameters):
  logL = -eval_log_expression(likelihood, parameters.get)
  logL0 = -eval_log_expression(likelihood, (lambda _ : 0.5))
  sys.stderr.write('likelihood gain is {:5.3f}-{:5.3f}={:5.3f}\n'
      .format(logL0, logL, logL0-logL))
  sys.stderr.flush()


def optimize_with_gradient_ascent(
    likelihood,
    iterations,
    initial_step,
    alpha,
    initial_parameters):
  parameters = dict(initial_parameters)
  step = initial_step
  for _ in range(iterations):
    gradient = compute_gradient(likelihood, parameters)
    parameters = update_parameters(parameters, gradient, step)
    step *= alpha
    estimate_likelihood_quality(likelihood, parameters)
    if False: # DBG
      print('step', step)
      print('parameters',sorted(parameters.items()))
  return parameters


def optimize_one_probability(likelihood, parameters, X):
  def f(x):
    parameters[X] = x[0]
    return -eval_log_expression(likelihood, parameters.get)
  def f_der(x):
    parameters[X] = x[0]
    # np.array seems needed although not documented (currently)
    return np.array([-eval_dx_log_expression(likelihood, parameters.get, X)])
  epsilon = 1e-15
  def accept(**kwargs):
    return all((epsilon <= x and x <= 1-epsilon) for x in kwargs['x_new'])
  minimizer_kwargs =\
    { 'method' : 'L-BFGS-B'
    , 'jac' : f_der
    , 'bounds' : [(epsilon, 1-epsilon)] }
  res = basinhopping(f, [parameters[X]],
      minimizer_kwargs=minimizer_kwargs,
      accept_test=accept,
      niter_success=5)
  if False: # DBG
    print('OPTIMIZATION_RESULT',res)
  return (res.x[0], res.fun)


# Cycles several times (iterations) through all coordinates. For each
# coordinate, it optimizes using a generic solver.
def optimize_with_coordinate_ascent(likelihood, iterations, initial_parameters):
  def evaluate_nonp(p, parameters):
    def assignment(x):
      assert x in parameters
      if x == p:
        return None
      return parameters[x]
    return partial_eval_expression(likelihood, assignment)
  parameters = dict(initial_parameters)
  old_value = None
  for _ in range(iterations):
    for p in parameters.keys():
      likelihood_p = evaluate_nonp(p, parameters)
      (new_p, value) = optimize_one_probability(likelihood_p, parameters, p)
      parameters[p] = new_p
      if False: # DBG
        print('param[',p,']=',new_p,'logL=',value)
    estimate_likelihood_quality(likelihood, parameters)
    if old_value is not None and abs(old_value - value) < 1e-9:
      break
    old_value = value
  return parameters


# }}}
# model helpers {{{
def save_parameters(out, parameters):
  ps = sorted((v, k) for k, v in parameters.items())
  for v, k in ps[::-1]:
    out.write('{:12.010f} {}\n'.format(v, k))


def load_parameters(infile):
  parameters = {}
  for line in infile:
      ws = line.split()
      parameters[ws[1]] = float(ws[0])
  return parameters

# }}}
# main {{{

def main():
  sys.stdout.write('''\
Instead of this script, use:
  sample_provenance.py
  compute_likelihood.py
  optimize_likelihood.py
  evaluate_likelihood.py
''')

if __name__ == '__main__':
  main()

# }}}
