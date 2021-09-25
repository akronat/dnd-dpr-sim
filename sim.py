#import sys
#import numpy as np
#import matplotlib.pyplot as plt
import re
import time
import argparse
import multiprocessing
import math
import statistics
import traceback
import fnmatch
from functools import reduce
from random import randrange
from collections import namedtuple
import itertools

def flatten_list(lst):
  return list(itertools.chain.from_iterable(lst))

# These are defaults, but can override with cmdline args
SETTINGS = {
  "iterations": 1000,
  "ac_values": ['14,16,18'],
  "levels": ['0-20'],
  "col_sort": ['lvl', 'avg'],
  "num_procs": 0,
  "repstats": ["mean", "sd%"], #["mean", "stdev", "sd%"],
  "fill_extra_sims": "pb",
  "show_detail": False,
}

REPSTAT_VAL_FMTS = {
  "mean": "{0:4.1f}",
  "stdev": "{0:4.1f}",
  "sd%": "{0:3.0f}%",
}
REPSTAT_HEAD_FMTS = {
  "mean": "{0:>4}",
  "stdev": "{0:>4}",
  "sd%": "{0:>4}",
}

PARSED_SIMS = {} # This is used due to multiprocessing instances

class Logger:
  def __init__(self, level):
    self.level = level
    self._lastProg = ''
    self._lastTemp = ''
    
  def _clearLine(self):
    if self._lastTemp or self._lastProg:
      print('\r' + ' ' * max(len(self._lastTemp), len(self._lastProg)) + '\r', end='', flush=True)
    self._lastProg = ''
    self._lastTemp = ''
  
  def _log(self, msg, *params, level=0):
    if self.level >= level:
      if self._lastTemp:
        self._clearLine()
      if self._lastProg:
        print()
      print(str(msg).format(*params))
      if (params and isinstance(params[-1], Exception)):
        traceback.print_tb(params[-1].__traceback__)
        #traceback.print_exc()

  def prog(self, msg, *params, level=0):
    if self.level >= level:
      self._clearLine()
      self._lastProg = str(msg).format(*params)
      print(self._lastProg, end='', flush=True)

  def temp(self, msg, *params, level=0):
    if self.level >= level:
      self._clearLine()
      self._lastTemp = str(msg).format(*params)
      print(self._lastTemp, end='', flush=True)

  def info(self, msg, *params):
    self._log(msg, *params, level=0)

  def detail(self, msg, *params):
    self._log(msg, *params, level=1)

  def debug(self, msg, *params):
    self._log(msg, *params, level=2)

  def trace(self, msg, *params):
    self._log(msg, *params, level=3)
log = Logger(0)

def roll(die):
  return randrange(1, die + 1)

def prof_bonus(level):
  return int((7 + level) / 4)

class Simulation:
  def __init__(self, name, level, sim_def, root_expr):
    self.name = name
    self.level = level
    self.sim_def = sim_def
    self.root_expr = root_expr
    self.conf = {}
    self.reg = {}
    self.root_expr.set_config_and_reg(self.conf, self.reg)
  
  def id(self):
    return f'{self.name}@{self.level}'

  def run_once(self):
    self.reg.clear()
    return self.root_expr.eval(State(False))

  def set_config(self, conf):
    self.conf.clear()
    self.conf.update(conf)

  def raw_expr(self):
    return self.root_expr.expr
    
State = namedtuple('State', 'crit')

class Expression(object):
  def __init__(self, type_name, expr, sub_exprs, evalFunc, min=None, max=None):
    num_subs = len(sub_exprs)
    if (min and num_subs < min) or (max and num_subs > max):
      raise Exception(f'Invalid number of parts in {expr}: expected between {min} and {max} but got {num_subs}')
    self.type_name = type_name
    self.expr = expr
    self.exprs = sub_exprs
    self.evalFunc = evalFunc
    self.conf = {}
    self.reg = {}
    self.props = {}

  def set_config_and_reg(self, conf, reg):
    self.conf = conf
    self.reg = reg
    for expr in self.exprs:
      if isinstance(expr, Expression):
        expr.set_config_and_reg(conf, reg)

  def eval(self, state):
    if self.evalFunc:
      return self.evalFunc(self, state)
    raise Exception('Invalid Expression with neither a func nor overidden eval')

class CreateExpr(object):
  def __init__(self, type_name, regex, min, max, parseFunc, evalFunc):
    self.type_name = type_name
    self.regex = re.compile(regex)
    self.parseFunc = parseFunc
    self.args = [evalFunc, min, max]

  def __call__(self, expr, exprs):
    result = Expression(self.type_name, expr, exprs, *self.args)
    if self.parseFunc:
      if len(exprs) > 1:
        m = self.regex.match(expr, len(exprs[0].expr))
      else:
        m = self.regex.match(expr)
      pr = self.parseFunc(result, m)
      if pr:
        for key in pr:
          setattr(result, key, pr[key])
    return result


def regex_match_at(regex, str, index):
  m = regex.match(str, index)
  return m and m.group(0)
def append_chunk(result, chunk, resets):
  if resets == 1 and chunk.startswith('(') and chunk.endswith(')'):
    chunk = chunk[1:-1]
  result.append(chunk)
def split_expr(expr, op_regex):
  result = []
  braces = 0
  currentChunk = ''
  chunkParensResets = 0
  op_matches = 0
  i = 0
  while i < len(expr):
    curChar = expr[i]
    if curChar == '(':
      braces += 1
    elif curChar == ')':
      braces -= 1
      if braces == 0:
        chunkParensResets += 1
    op_match = regex_match_at(op_regex, expr, i)
    if braces == 0 and op_match:
        append_chunk(result, currentChunk, chunkParensResets)
        currentChunk = ''
        chunkParensResets = 0
        i += len(op_match)
    else:
      currentChunk += curChar
      i += 1
  if (braces != 0):
    raise Exception(f'Unbalanced parentheses in expression "{expr}"')
  append_chunk(result, currentChunk, chunkParensResets)
  return (result, chunkParensResets)

def check_type(expr, type_names):
  if not expr.type_name in type_names:
    raise Exception(f'Type of "{expr.expr}" was expected to be one of "{type_names}", but was "{expr.type_name}"')
  return expr.type_name

def expr_assign_eval(ctx, state):
  if ctx.stored_type == 'function':
    ctx.reg[ctx.exprs[0].funcname] = ctx.exprs[1]
  elif ctx.stored_type == 'variable':
    ctx.reg[ctx.exprs[0].varname] = ctx.exprs[1].eval(state)
  return 0

def expr_attack_eval(ctx, state):
  droll = roll(20)
  if ctx.vantage == 'adv':
    droll = max(droll, roll(20))
  elif ctx.vantage == 'dis':
    droll = min(droll, roll(20))

  if droll >= ctx.critmin:
    state = state._replace(crit=True)

  if state.crit or (ctx.exprs[0].eval(state) + droll >= ctx.conf['ac']):
    return ctx.exprs[1].eval(state)
  return 0

def expr_save_eval(ctx, state):
  droll = roll(20)
  if ctx.vantage == 'adv':
    droll = max(droll, roll(20))
  elif ctx.vantage == 'dis':
    droll = min(droll, roll(20))

  dmg = ctx.exprs[1].eval(state)
  if droll + ctx.conf['sm'] >= ctx.exprs[0].eval(state):
    return math.floor(dmg * ctx.failmod)
  else:
    return dmg

def expr_reroll_lte_eval(ctx, state):
  droll = ctx.exprs[0].eval(state)
  if droll <= ctx.exprs[1].eval(state):
    return ctx.exprs[0].eval(state)
  return droll

SPLIT_EXPRS = [
  CreateExpr('assign', r'\:\=', 2, 2,
    lambda ctx, m: { 'stored_type': check_type(ctx.exprs[0], ['variable', 'function']) },
    lambda ctx, s: expr_assign_eval(ctx, s),
  ),
  CreateExpr('check', r'\=\>', 2, 2, None, lambda ctx, s: ctx.exprs[1].eval(s) if ctx.exprs[0].eval(s) else 0 ),
  CreateExpr('attack', r'\=atk(?:\:(\d+))?(?:\:(adv|dis))?\>', 2, 2,
    lambda ctx, m: { 'critmin': int(m.group(1) or 20), 'vantage': m.group(2) },
    expr_attack_eval
  ),
  CreateExpr('save', r'\=sav\:(\d+)(?:\:(adv|dis))?\>', 2, 2,
    lambda ctx, m: { 'failmod': (float(m.group(1)) / 100.0) or 0.5, 'vantage': m.group(2) },
    expr_save_eval
  ),
  CreateExpr('gte', r'\>\=', 2, 2, None, lambda ctx, s: ctx.exprs[0].eval(s) >= ctx.exprs[1].eval(s) ),
  CreateExpr('lte', r'\<\=', 2, 2, None, lambda ctx, s: ctx.exprs[0].eval(s) <= ctx.exprs[1].eval(s) ),
  CreateExpr('add', r'\+', 2, None, None, lambda ctx, s: sum([e.eval(s) for e in ctx.exprs]) ),
  CreateExpr('sub', r'\-(?!\>)', 2, None, None, lambda ctx, s: ctx.exprs[0].eval(s) - sum([e.eval(s) for e in ctx.exprs[1:]]) ),
  CreateExpr('mul', r'\*', 2, None, None, lambda ctx, s: reduce(lambda a,e: a * e.eval(s), ctx.exprs, 1) ),
  CreateExpr('div', r'/', 2, 2, None, lambda ctx, s: math.floor(ctx.exprs[0].eval(s) / ctx.exprs[1].eval(s)) ),
  CreateExpr('repeat', r'\#', 2, 2, None, lambda ctx, s: sum([ctx.exprs[1].eval(s) for i in range(int(ctx.exprs[0].eval(s)))]) ),
  CreateExpr('reroll_lte', r'\@rrlte:', 2, 2, None, expr_reroll_lte_eval ),
]
VALUE_EXPRS = [
  CreateExpr('number', r'^\d+$', 1, 1, None, lambda ctx, s: float(ctx.expr) ),
  CreateExpr('roll', r'^(\d*)d(\d+)(?:k(h|l)(\d+))?$', 1, 1,
    lambda ctx, m: {
      'dice': [int(m.group(2))] * int(m.group(1) or 1),
      'keep': (-1 if m.group(3) == 'l' else 1) * int(m.group(4) or m.group(1) or 1),
    },
    lambda ctx, s: sum(sorted([roll(d) for d in ctx.dice], reverse=(ctx.keep > 0))[:(abs(ctx.keep or len(ctx.dice)))]),
  ),
  CreateExpr('roll', r'^(\d*)D(\d+)?$', 1, 1,
    lambda ctx, m: { 'dice': [int(m.group(2))] * int(m.group(1) or 1) },
    lambda ctx, s: sum([roll(d) + roll(d)*s.crit for d in ctx.dice]),
  ),
  CreateExpr('armor_class', r'^AC$', 1, 1, None, lambda ctx, s: ctx.conf['ac'] ),
  CreateExpr('prof_bonus', r'^PB$', 1, 1, None, lambda ctx, s: ctx.conf['pb'] ),
  CreateExpr('level', r'^LV$', 1, 1, None, lambda ctx, s: ctx.conf['lv'] ),
  CreateExpr('crit_mult', r'^CM$', 1, 1, None, lambda ctx, s: s.crit * 1 + 1 ),
  CreateExpr('crit_binary', r'^CB$', 1, 1, None, lambda ctx, s: s.crit * 1 ),
  CreateExpr('empty', r'^$', 1, 1, None, lambda ctx, s: 0 ),
  CreateExpr('variable', r'^\$([\d\w]+)$', 1, 1,
    lambda ctx, m: { 'varname': m.group(1) },
    lambda ctx, s: ctx.reg.get(ctx.varname, 0),
  ),
  CreateExpr('function', r'^\!([\d\w]+)$', 1, 1,
    lambda ctx, m: { 'funcname': m.group(1) },
    lambda ctx, s: ctx.reg.get(ctx.funcname, lambda s: 0).eval(s),
  ),
]

def parse_sim_expr(expr):
  for expr_ctr in SPLIT_EXPRS:
    (parts, parensGroups) = split_expr(expr, expr_ctr.regex)
    if len(parts) == 1 and parensGroups == 1 and expr.startswith('(') and expr.endswith(')'):
      return parse_sim_expr(parts[0])
    if len(parts) > 1:
      return expr_ctr(expr, [parse_sim_expr(p) for p in parts])
  
  for expr_ctr in VALUE_EXPRS:
    if expr_ctr.regex.match(expr):
      return expr_ctr(expr, [expr])
  raise Exception(f'Invalid expression: "{expr}"')

def parse_sim_def(sim_def):
  try:
    name, expr = sim_def.split(':', 1)
    levels = [0]
    if '@' in name:
      name, levels_str = name.split('@')
      levels = parse_ranges([levels_str])
    sims = []
    for level in levels:
      sim = Simulation(name, int(level), sim_def, parse_sim_expr(expr.replace(' ', '')))
      # Perform a test run, to make sure it's all good!
      sim.set_config({ 'ac': 10, 'pb': 2, 'lv': 1, 'sm': 0 })
      sim.run_once()
      sims.append(sim)
    return sims
  except Exception as e:
    log.info(f'Failed to parse simulation definition "{sim_def}": {e}', e)
    return []


def parse_sims(sim_defs):
  return list(flatten_list([parse_sim_def(sd) for sd in sim_defs]))


def init_parsed_sims(sim_defs):
  PARSED_SIMS.clear()
  sims = parse_sims(sim_defs)

  # Add missing sims for PB breakpoints (just copy next highest sim)
  if SETTINGS["fill_extra_sims"] != "none":
    fill_all_levels = SETTINGS["fill_extra_sims"] == "all"
    sim_nl = sims_by_name_level(sims)#, SETTINGS['levels'])
    extra_pb_sims = []
    for name, levels in sim_nl.items():
      if not levels:
        continue
      min_lvl = min(levels.keys())
      max_level = max(levels.keys())
      for i in range(min_lvl, 21):
        is_pb_breakpoint = prof_bonus(i) != prof_bonus(i - 1)
        if (i not in levels) and (fill_all_levels or is_pb_breakpoint):
          for j in range(i - 1, min_lvl - 1, -1):
            if j in levels and j > 0 and i < max_level:
              sim_def = levels[j].raw_expr()
              extra_pb_sims.append(f'{name}@{i}:{sim_def}')
              break
    sims += parse_sims(extra_pb_sims)

  for sim in sims:
    PARSED_SIMS[sim.id()] = sim
    



def list_simulations(sims):
  names = [f'{s.name}@{s.level}' for s in sims]
  max_len = max(*[len(n) for n in names])
  for s in sims:
    log.info(f'{s.name}@{s.level}:'.ljust(max_len + 2) + s.raw_expr())

def sim_num(sim_id, config, num):
  sim = PARSED_SIMS[sim_id]
  sim.set_config(config)
  data = []
  for _ in range(num):
    data.append(sim.run_once())
  result = {}
  result["mean"] = statistics.mean(data)
  result["stdev"] = statistics.stdev(data)
  return result

def reduce_wo_stats(works):
  result = {}
  result["mean"] = sum([wo["mean"] for wo in works]) / len(works)
  #TODO: How to do this correctly!?
  #result["stdev"] = math.sqrt(sum([wo["stdev"] ** 2 for wo in works]))
  result["stdev"] = math.sqrt(sum([wo["stdev"] ** 2 for wo in works]) / len(works))
  result["sd%"] = 100.0 * result["stdev"] / abs(result["mean"]) if result["mean"] else math.nan
  return result

def calc_savemod_from_ac(ac):
  return ac - 10

def run_sim_for_levels_acs(name, level_sims, mp_pool):
  log.info("Simulating {}", name)
  iters = SETTINGS["iterations"]
  levels = sorted(level_sims.keys())
  results = {}
  for level in levels:
    results[level] = {}
    sim = level_sims[level]
    for ac in SETTINGS["ac_values"]:
      log.temp(
        "Simulating {} at level {}, against ac {}, {} iterations",
        name, level, ac, SETTINGS["iterations"]
      )
      
      config = { 'ac': ac, 'pb': prof_bonus(level), 'lv': level, 'sm': calc_savemod_from_ac(ac) }
      
      # Now run the sim, with specified number of processes
      nw = SETTINGS["num_procs"]
      if not mp_pool or iters < nw:
        work_outs = [sim_num(sim.id(), config, iters)]
      else:
        sim_args = [[sim.id(), config, iters // nw] for i in range(nw)]
        work_outs = mp_pool.starmap(sim_num, sim_args)
      results[level][ac] = reduce_wo_stats(work_outs)
  return ( name, results )

def create_worker_pool(num, sims):
  return multiprocessing.Pool(num, init_parsed_sims, [[s.sim_def for s in sims]])

def parse_ranges(str_ranges):
  '''Parse strings such as "1-5,7,9,11-20" into a list of (integer) values'''
  values = []
  ranges = flatten_list([str_rng.split(',') for str_rng in str_ranges])
  for rng in ranges:
    if "-" in rng:
      minlvl, maxlvl = rng.split('-')
      values.extend(range(int(minlvl), int(maxlvl) + 1))
    else:
      values.append(int(rng))
  return sorted(list(set(values)))

def sims_by_name_level(sims, level_specs=None):
  levels = range(0, 21)
  if level_specs:
    levels = parse_ranges(level_specs)
  nl_sims = {}
  for sim in sims:
    if not sim.name in nl_sims:
      nl_sims[sim.name] = {}
    if (sim.level in levels):
      nl_sims[sim.name][sim.level] = sim
  return nl_sims

def pad_col(index, value, col_widths):
  if index <= 1 or (SETTINGS["show_detail"] and index == (len(col_widths) - 1)):
    return value.ljust(col_widths[index])
  return value.rjust(col_widths[index])

def run_simulations(sims):
  repstats = SETTINGS["repstats"]
  
  nl_sims = sims_by_name_level(sims, SETTINGS['levels'])

  # Perform Simulation
  log.info("Performing {} iterations of each simulation", SETTINGS["iterations"])
  start_time = time.time()
  num_procs = SETTINGS["num_procs"]
  if num_procs <= 1:
    all_results = [run_sim_for_levels_acs(name, nl_sims[name], None) for name in nl_sims]
  else:
    with create_worker_pool(num_procs, sims) as mp_pool:
      all_results = [run_sim_for_levels_acs(name, nl_sims[name], mp_pool) for name in nl_sims]
  end_time = time.time()
  duration = (end_time - start_time)
  log.info('Simulations completed in {0:0.2f} seconds', duration)
  
  # Collate Result Rows
  rows = []
  for name, res in all_results:
    for level in res:
      acvals = res[level]
      stats_for_acs = [acvals[ac] for ac in sorted(acvals.keys())]
      vals = [[sv[sk] for sk in repstats] for sv in stats_for_acs]
      avgs = [sum([v[i] for v in vals]) / len(vals) for i in range(len(repstats))]
      rows.append([name, level] + vals + [avgs])
  
  # Prepare Header Row
  hrow0 = ["Name", "lvl"]\
          + ["{0:2.0f}AC/+{1}SM".format(x, calc_savemod_from_ac(x)) for x in SETTINGS["ac_values"]]\
          + ["Average"]
  num_acs = len(SETTINGS["ac_values"])
  hrow1 = ['','',] + [' '.join([REPSTAT_HEAD_FMTS[s].format(s) for s in repstats]) for i in range(num_acs + 1)]
  if SETTINGS["show_detail"]:
    hrow0 += ['Simulation expression']
    hrow1 += ['']
  
  # Sort Result Rows
  col_sort = SETTINGS["col_sort"]
  trmheads = [h.strip() for h in hrow0]
  cinds = [trmheads.index(c) for c in col_sort if c in trmheads]
  rows = sorted(rows, key=lambda r: [r[i] for i in cinds] + [r[-1]] + r)
  
  # Prepare Result Rows
  drows = []
  for i, row in enumerate(rows):
    cols = [row[0], str(row[1] or '-')]
    cols += [" ".join([REPSTAT_VAL_FMTS[repstats[j]].format(s) for j, s in enumerate(v)]) for v in row[2:]]
    if SETTINGS["show_detail"]:
      sim = PARSED_SIMS.get(f'{row[0]}@{row[1]}', None)
      cols += [sim.raw_expr() if sim else '']
    drows.append(cols)
  
  # Determine column widths
  col_widths = [0] * len(hrow0)
  for row in [hrow0, hrow1] + drows:
    for i, col in enumerate(row):
      col_widths[i] = max(col_widths[i], len(col))

  # Print Table
  log.info('') # Empty line before the table output
  header = " | ".join([pad_col(i, s, col_widths) for i, s in enumerate(hrow0)])
  log.info(header)
  log.info(" | ".join([pad_col(i, s, col_widths) for i, s in enumerate(hrow1)]))
  log.info("-" * len(header))
  
  for i, row in enumerate(drows):
    # Print a separator line between levels
    if i > 0 and cinds[0] == 1 and row[1] != drows[i - 1][1]:
      log.info("-" * len(header))
    log.info(" | ".join([pad_col(i, s, col_widths) for i, s in enumerate(row)]))


def parse_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="""D&D Attack Simulator
Performs simulations of attacks based on custom definitions. Simulations are
defined with the following form:
  <name>@<level>: <expression>
Where the expression is a custom defintion such as in the following:
  GreatAxe@1: 3+PB =atk> 1D12+3
  GreatSword@1: 3+PB =atk> 2D6+3
  Dual Wield@1: (3+PB =atk> 1D6+3) + (3+PB =atk> 1D6)
  GreatSword with GreatWeaponMaster@1: 3+PB-5 =atk> 2D6+3+10
In the above examples, 3 is the attacking attribute modifier, while PB is the
automatically calculated proficiency bonus. The uppercase D in the 1D6 damage
roll means that the number of dice is doubled on a crit.
Extra Attack can be easily defined  with # notation:
  GreatAxe@5: 2#(4+PB =atk> 1D12+3)
  GreatSword@5: 2#(4+PB =atk> 2D6+3)
  Dual Wield@5: 2#(4+PB =atk> 1D6+3) + (3+PB =atk> 1D6)
More advanced definitions are also possible...
  GreatAxe with GreatWeaponFighting@1: 3+PB =atk> CM#(1d12 @rrlte: 2) + 3
  GreatSword with GreatWeaponFighting@1: 3+PB =atk> CM#(2#(1d6 @rrlte: 2)) + 3
  GreatAxe with Advantage@1: 3+PB =atk:adv> 1D12+3
  GreatAxe with Disadvantage@1: 3+PB =atk:dis> 1D12+3
  GreatAxe that crits on a 19@1: 3+PB =atk:19> 1D12+3
  GreatAxe with Advantage that crits on a 19@1: 3+PB =atk:19:adv> 1D12+3
  Dual Wield with a Sneak Attack@1: ($a := (3+PB =atk> 1D6+3 + 1D6)) + (3+PB =atk> 1D6 + ($a<=0 => 1D6)) + $a
""")
  # parser.add_argument('-l', '--list', action="store_true",
  #   help='List the available simulations.')
  parser.add_argument('-d', '--detail', action="store_true",
    help='Show the formula for the sim in the output table.')
  parser.add_argument('-e', '--extra', default=SETTINGS["fill_extra_sims"], choices=['none', 'pb', 'all'],
    help='''How to handle missing levels for sims:
 - none: don\'t add any.
 - pb: only add proficiency bonus breakpoint levels.
 - all: fill in all missing levels.
 (Level 0 sims are never extended)''')
  parser.add_argument('-p', '--parallel', type=int, default=SETTINGS["num_procs"],
    help='Number of parallel processes to run. Defaults to number of CPU cores.')
  parser.add_argument('-i', '--iterations', type=int, default=SETTINGS["iterations"],
    help='Number of iterations to perform for each simulation.')
  parser.add_argument('-m', '--sims', nargs="*", default=None,
    help='The filter simulations (in files) via glob syntax.')
  parser.add_argument('-c', '--custom', nargs="*", default=[],
    help=f'The custom simulation definitions to perform.')
  parser.add_argument('-f', '--files', nargs="+", default=[],
    help=f'The files with custom simulations (one per line) to perform.')
  parser.add_argument('-l', '--levels', nargs="+", default=SETTINGS["levels"],
    help='The levels to simulate (e.g. "1-5,9,13")')
  parser.add_argument('-a', '--acvalues', nargs="+", default=SETTINGS["ac_values"],
    help='The AC values to test against (e.g. "12,14,16-20,22").')
  parser.add_argument('-s', '--sort', nargs="+", default=SETTINGS["col_sort"],
    help='Column(s) to sort by, comma separated.')
  parser.add_argument('-t', '--stats', nargs="+", default=SETTINGS["repstats"],
    choices=['mean', 'stdev', 'sd%'], help='Statistics to report.')
  parser.add_argument('-v', '--verbose', default=0, action='count',
    help='The verbosity of the output; -vvv for maximum verbosity')
  return parser.parse_args()


if __name__ == '__main__':
  multiprocessing.freeze_support()
  args = parse_args()
  log.level = args.verbose
  SETTINGS["repstats"] = args.stats
  SETTINGS["fill_extra_sims"] = args.extra
  SETTINGS["num_procs"] = args.parallel or multiprocessing.cpu_count()
  SETTINGS["iterations"] = args.iterations
  SETTINGS["ac_values"] = sorted(parse_ranges(args.acvalues))
  SETTINGS["levels"] = args.levels
  SETTINGS["col_sort"] = args.sort
  SETTINGS["show_detail"] = args.detail

  sims = []
  if args.files:
    for fname in args.files:
      with open(fname) as f:
        sims += [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
  if args.sims is not None:
    unfiltered_sims = sims
    sims = []
    for sim_def in unfiltered_sims:
      sim_name = sim_def.split('@', 1)[0]
      for pattern in args.sims:
        if fnmatch.fnmatch(sim_name, pattern):
          sims.append(sim_def)
          break
  if args.custom:
    sims += args.custom
  if sims:
    init_parsed_sims(sims)
    # if args.list:
    #   list_simulations(PARSED_SIMS.values())
    # else:
    run_simulations(PARSED_SIMS.values())
  else:
    log.info('No simulations to run!')
