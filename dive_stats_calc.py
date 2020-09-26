import numpy as np
import sys
sys.path.insert(1,"PySCIPOpt/src")
import pyscipopt as scip
import pandas as pd 
from itertools import chain
import math


def get_agg_stats(n_cands, diving_state=None, lp_names=None, lp_vals=None, 
					root_tuple = None, std_method='minmax', stats='all'):
	'''
	Gets an average of all statistics
	Params
	-------------
	diving_state : Dict
		Scip diving stats
	lp_names and lp_vals:
		Names of scip variables and current values
	root_tuple : Tuple
		List of variable names at root
			and list of all solution values at the root node
	std_method : Str
		Method for standardization
	stats : Str
		Which stats should we use
	'''
	if stats == 'all':
		stat_names = ['fract', 'coef', 'pseudo', 'linesearch']
	else:
		stat_names = stats
	
	stat_lists = np.empty((len(stat_names), n_cands))
	curr_pos = 0

	if 'fract' in stat_names:
		fract_stats = get_fract_diving_stats(lp_vals)
		stat_lists[curr_pos,:] = np.array(fract_stats)
		curr_pos += 1
	
	if 'coef' in stat_names:
		_ ,coef_stats = get_coef_diving_stats(diving_state)
		if std_method == 'minmax' and len(set(coef_stats)) > 1:
			coef_stats = (coef_stats-min(coef_stats))/(max(coef_stats)-min(coef_stats))
		stat_lists[curr_pos,:] = np.array(coef_stats)
		curr_pos += 1
	
	if 'pseudo' in stat_names:
		pseudo_stats = get_pseudo_diving_stats(diving_state)
		if std_method == 'minmax' and len(set(pseudo_stats)) > 1:
			pseudo_stats = (pseudo_stats-min(pseudo_stats))/(max(pseudo_stats)-min(pseudo_stats))
		stat_lists[curr_pos,:] = np.array(pseudo_stats)
		curr_pos += 1

	if 'linesearch':
		line_stats = get_linesearch_diving_stats(lp_names, lp_vals, root_tuple)
		if std_method == 'minmax' and len(set(line_stats)) > 1:
			line_stats = (line_stats-min(line_stats))/(max(line_stats)-min(line_stats))
		stat_lists[curr_pos,:] = np.array(line_stats)
		curr_pos += 1

	return np.average(stat_lists,axis=0)



def get_fract_diving_stats(fract_vals):
	'''
	Gets fractionality diving stats for a particular node
	Here for stats, we would branch on min value, since it is the closest from whole number
	
	Params
	-------------
	fract_vale : List
		List of how far we are from whole number. Contains all values (int and float)
	'''
	# How far we are from 1
	fract_stats_up = np.array([x if x < 0.5 else 1-x for x in fract_vals])
	# Length of 2x nb candidates
	fract_stats = [val for pair in zip(1-fract_stats_up, fract_stats_up) for val in pair]

	return np.array(fract_stats)


def get_coef_diving_stats(diving_state):
	'''
	Gets coefficient diving stats for particular node
	Min also the best (except for 0)

	Params
	------------
	diving_state : Dict
		Stats information used for dives
	'''
	n_lock_up = diving_state['n_locks_up']
	n_lock_down = diving_state['n_locks_down']
	
	coef_stats = [val for pair in zip(n_lock_down, n_lock_up) for val in pair]

	max_val = max(chain(n_lock_up,n_lock_down))
	# Need to get min for each, which is not 0 when diving
	coef_stats_diving = list(map(lambda x: x if x>0 else max_val, coef_stats))

	return np.array(coef_stats), np.array(coef_stats_diving)


def get_pseudo_diving_stats(diving_state):
	'''
	Gets pseudocost diving stats for particular node

	Params
	------------
	diving_state : Dict
		Stats information used for dives
	'''
	ps_up = diving_state['ps_up']
	ps_dwn = diving_state['ps_down']

	pseudo_stats = [val for pair in zip(ps_dwn, ps_up) for val in pair]

	return np.array(pseudo_stats)


def get_linesearch_diving_stats(lp_names, lp_vals, root_tuple):
	'''
	Gets linesearch diving stats for particular node
	Branching on smallest ratio

	Params
	------------
	lp_names and lp_vals:
		Names of scip variables and current values
	root_tuple : Tuple
		List of variable names at root
			and list of all solution values at the root node
	'''
	def linesearch_calc(curr_val, root_val, max_val = None):
		'''
		Makes calculations for linesearch map function
		'''
		# If curr val < root_val :
		# 	 			  curr_val - floor(curr_val) 
		#    Dist ratio = --------------------------
		# 				    root_val - curr_val
		# Elif curr val > root_val :
		# 	             ceil(curr_val) - curr_val
		#   Dist ratio = --------------------------
		# 				    curr_val - root_val
		if curr_val == root_val:
			dist_ratios = (max_val, max_val)
		elif curr_val > root_val:
			dist_ratio = (math.ceil(curr_val) - curr_val) / (curr_val - root_val)
			dist_ratios = (max_val, dist_ratio)
		else:
			dist_ratio = (curr_val - math.floor(curr_val)) / (root_val - curr_val)
			dist_ratios = (dist_ratio, max_val)

		return dist_ratios

	def get_lp_vals_currandroot(curr_names, root_names, root_vals, default_val=1):
		'''
		Get the root values for all current lp candidates
		'''
		matching_root_vals = []
		for c_name in curr_names:
			if c_name in root_names:
				matching_root_vals.append(root_vals[root_names.index(c_name)])
			else:
				matching_root_vals.append(default_val)
		return matching_root_vals


	# Get all fract nodes which are also fract at root node
	root_names, root_vals = root_tuple

	matching_root_vals = get_lp_vals_currandroot(lp_names, root_names, root_vals)
	max_val = np.nanmax(np.array(matching_root_vals, dtype=np.float64))
	
	dist_ratios = [val for pair in list(map(linesearch_calc, lp_vals, matching_root_vals)) for val in pair]
	# Replace None by max value in list
	dist_ratios = [max_val if i is None else i for i in dist_ratios]

	return np.array(dist_ratios)


def get_vectlength_diving_stats():
	'''
	Gets vector length diving stats for particular node

	Params
	------------
	
	'''

	pass



