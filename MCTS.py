import utilities
import math
import multiprocessing as mp
import sys
# sys.path.append("..")
import numpy as np
import time
import re
import random
from multiprocessing import Process, Value, Manager
import faulthandler
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import pyscipopt as scip
from heapq import nsmallest
import pandas as pd
import os
import datetime

class Tree():
	''' 
	Simple tree which will be used MCST

	Parameters
	-------------
	problem_name : String
		Name of the linear program to be solved
	seed : Int
		RNG seed
	brancher : Branching Policy
		PolicyBranching(scip.Branchrule)
	adding_new_nodes : String (all, best, random)
		How should new nodes be added
	'''

	def __init__(self, seed, brancher, adding_new_nodes='all', branches_per_lp_solve=5, exp_ratio=np.sqrt(2) ):
		''' 
		Only starts with root node.
		'''

		self.brancher = brancher

		self.branches_per_lp_solve = branches_per_lp_solve

		self.root_node = None

		# Focus node
		self.curr_node = None

		# Phase in ['rollout', 'add_node', 'dive', 'add_root_node']
		# Starts at dive and since we only have root node and no rollout is needed
		self.phase = 'add_root_node'

		self.adding_new_nodes = adding_new_nodes

		self.scip_has_branched = False

		self.start_var = True

		# Rollout stats
		self.rollout_nb = 1
		self.max_rollout_nb = 4

		self.rollout_sols = np.ones(self.max_rollout_nb) * np.inf
		# Those are the parent of the nodes which hae a dive performed on, since those nodes may not be created
		self.rollout_tree_nodes = []

		self.best_sol_list = []
		self.nb_lp_solves = []

		self.curr_best_sol = 1e8

		self.ndomchgs = 0
		self.ndomchgs_graph = 0

		self.update_graph = True

		self.exp_ratio = exp_ratio

		# To change root of tree to not leaf. Only done once
		self.first_branch = True

		# List of all available nodes
		self.nodes = []

		self.created_node = False

		self.mt_second_depth = False
		self.branching_to_add_cands = False


	def get_best_sol(self):
		''' 
		Returns best solution from tree
		'''
		return self.root_node.best_lp_sol


	def update_root(self,node):
		''' 
		Adds root node to tree
		'''
		self.root_node = node

	def get_root_stats(self):
		''' 
		Returns root nb visits and root nb wins
		'''
		root = self.get_root()
		return root.child_nb_wins, root.child_nb_visits

	def get_root(self):
		'''
		Returns root node of the tree
		'''
		return self.root_node

	def add_all_nodes(self):
		''' 
		Adds a new node for each node which had a dive performed on
		'''
		for node in self.rollout_tree_nodes:
			self.add_node(node)

	def add_best_node(self):
		''' 
		Adds a new node at the node where best dive was
		'''
		max_val = min(self.rollout_sols)
		max_idx = self.rollout_sols.tolist().index(max_val)
		node = self.rollout_tree_nodes[max_idx]

		# Add node and delete other node objects
		self.add_node(node)
		for n in np.delete(np.array(self.rollout_tree_nodes),[max_idx]):
			del n
		

	def add_rand_node(self):
		''' 
		Adds a new node randomly between the nodes which had dive performed on
		'''
		rand_idx = random.randint(0, self.max_rollout_nb-1)
		node = self.rollout_tree_nodes[rand_idx]

		# Add node and delete other node objects
		self.add_node(node)	
		for n in np.delete(np.array(self.rollout_tree_nodes),[rand_idx]):
			del n


	def add_node(self, node):
		'''
		Adds node as a new leaf of the tree.
		hot_start: Method for initializing nodes. None if random initialization
		'''
		node.parent.set_leaf_node()

		node.parent.add_children(node)
		self.add_node_to_list(node)


	def update_focus_node(self, cand, node_ind):
		''' 
		Updates current node in tree, as well as the index of it's parent
		'''
		self.curr_node = cand
		self.curr_node.ind_of_parent = node_ind


	def increment_to_root(self, node=None, wins=False, sol=None):
		'''
		Once we arrive at the end of the episode, we increment path back to root
		We first need to find the best node
		If all infeasible or all same sol, we choose randomly one of the sols
		node : Node to increment stats on. If wins, we don't have a node, since we will chose from list of rollout nodes
		wins : If true, we increment number of wins at node, otherwise, we increment nb runs
		'''
		# Update best lp sol at each node
		if sol is not None:
			node.update_best_sol(sol)

		parent_node = node.parent
		node_index = node.ind_of_parent
		# Extract validity of solution from state
		if wins:
			parent_node.child_nb_wins[node_index] += 1
			parent_node.nb_wins += 1
		else:
			parent_node.child_nb_visits[node_index] += 1
			parent_node.nb_visits += 1

		parent_node.update_ucb(self.exp_ratio)

		# Only increment if we are not at root
		if parent_node.parent is not None:
			self.increment_to_root(parent_node, wins, sol)
		# If parent is root
		elif sol is not None:
			parent_node.update_best_sol(sol)


	def get_graph_vals(self):
		'''
		Returns list of best feasible sols found and list of corresponding nb lp solves
		'''
		return self.best_sol_list, self.nb_lp_solves


	def get_root_ucb(self):
		'''
		Returns a list of root node ucb stats
		'''
		return self.root_node.child_ucb


	def get_best_dive_sol(self):
		'''
		Gets the best sol found in the first n dives
		Returns 1e8 otherwise
		'''
		return min(self.rollout_sols)

	def add_node_to_list(self, new_node):
		self.nodes.append(new_node)


class Node():
	'''
	Node of the tree

	Params
	---------
	parent : Node
		parent of current node, with None root node
	scip_node: Scip node
		Scip node associated with MCTS node. 
		None when at root
	ind_of_parent: int
		Index of parent's candidates. None if no parent node (root)
	branch_up: Int or None
		Wether node is branched up (0) or down (1)
		None when at root
	init_stats_up and init_stats_down: List or None
		List of stats which will be used as hot start of UCB stats
	'''

	def __init__(self, parent, scip_name = None, ind_of_parent=None, branch_up=None, init_stats_up = None, init_stats_down = None, cost_up=None, cost_down=None):

		self.parent = parent

		self.candidates = None
		# List of all nb_visits for candidates and wins
		self.cand_visits = None
		self.cand_wins = None

		self.scip_name = scip_name

		# Node is associated with left (down) branching
		self.branch_up = branch_up

		# Always initialized as true
		self.leaf_node = True

		# Child stats
		self.child_nb_visits = []
		self.child_nb_wins = []
		self.child_ucb = []
		# Current node stats
		self.nb_visits = 0
		self.nb_wins = 0
		
		self.best_lp_sol = 1e8

		# List of all nb_visits for candidates and wins
		self.cand_visits = None
		self.cand_wins = None

		self.dive_sol = 1e8

		self.ind_of_parent = ind_of_parent

		self.children = []
		# Before we add node (only useful if > 1 rollout before adding node)
		self.temp_children = []

		self.init_stats_up = init_stats_up
		self.init_stats_down = init_stats_down

		if self.parent is None:
			self.depth = 0
		else:
			self.depth = self.parent.depth + 1


	def get_parent(self):
		'''
		Returns a node's parent
		'''
		return self.parent


	def set_leaf_node(self, leaf_bool=False):
		'''
		Changes value of leaf node attribute
		'''
		self.leaf_node = leaf_bool


	def get_best_sol(self):
		'''
		Gets best feasible solution
		'''
		return self.root_node.best_lp_sol


	def update_ucb(self, exp_ratio=np.sqrt(2)):
		'''
		Calculate upper confidence bound
		'''
		if self.nb_visits == 1:
			self.child_ucb = (self.child_nb_wins / self.child_nb_visits) + \
								exp_ratio * np.sqrt(self.nb_visits/self.child_nb_visits)		
		else:
			self.child_ucb = (self.child_nb_wins / self.child_nb_visits) + \
								exp_ratio * np.sqrt(np.log(self.nb_visits)/self.child_nb_visits)


	def add_children(self, child):
		self.children.append(child)


	def set_ucb_stats(self, ucb_stats=None):
		'''
		Initialize nb visit, nb wins, ucb stats.
		Not in __init__ function since we don't have candidates at the start
		'''
		# Child stats
		self.child_nb_visits = np.ones(len(self.candidates)) * 1e-8
		self.child_nb_wins = np.zeros(len(self.candidates))

		if ucb_stats is None:
			# Initializes to all infinity if 
			if self.init_stats_down is None or (all(self.init_stats_down == 1) and all(self.init_stats_up == 1)):
				self.child_ucb = np.ones(len(self.child_nb_wins)) * np.inf
			else:
				self.child_ucb = [val for pair in zip(self.init_stats_up, self.init_stats_down) for val in pair]
		else:
			self.child_ucb = ucb_stats



	def update_best_sol(self, lp_sol):
		'''
		Updates best solution for current node
		'''
		if self.best_lp_sol > lp_sol:
			self.best_lp_sol = lp_sol


	def choose_candidate(self):
		'''
		Candidate is chosen based on balance between number of visits and "wins"
		returns both node, as well as direction as 0 for up, 1 for down, 
						and the index corresponding to the position of children in the list of the parent's candidate
		'''
		if len(self.child_ucb) == 0:
			cand, direction, max_ind = (None,None,None)
		else:
			max_val = max(self.child_ucb)
			max_ind = self.child_ucb.tolist().index(max_val)

			cand = self.candidates[max_ind]
			direction = max_ind % 2

		return cand, direction, max_ind


	def set_cands(self, cands):
		''' 
		Sets a list of candidates for node
		'''
		# Duplicates each element (first is up, second is down)
		self.candidates = [x for x in cands for _ in (0,1)]
		self.cand_visits = np.zeros(len(self.candidates))
		self.cand_wins = np.zeros(len(self.candidates))


	def is_leaf(self):
		''' 
		Returns true if node is node is leaf node
		'''
		return self.leaf_node


class SolvingStatsRecorder(scip.Eventhdlr):

	def __init__(self, solving_stats):
		self.solving_stats = solving_stats

	def eventinit(self):
		self.model.catchEvent(scip.SCIP_EVENTTYPE.NODEFEASIBLE, self)
		self.model.catchEvent(scip.SCIP_EVENTTYPE.NODEINFEASIBLE, self)
		self.model.catchEvent(scip.SCIP_EVENTTYPE.NODEBRANCHED, self)

	def eventexit(self):
		self.model.dropEvent(scip.SCIP_EVENTTYPE.NODEFEASIBLE, self)
		self.model.dropEvent(scip.SCIP_EVENTTYPE.NODEINFEASIBLE, self)
		self.model.dropEvent(scip.SCIP_EVENTTYPE.NODEBRANCHED, self)

	def eventexec(self, event):
		if len(self.solving_stats) < self.model.getNNodes():
			keys = ['nlps','primalbound']
			self.solving_stats.append({key: self.model.getSolvingStats()[key] for key in keys})
			# self.solving_stats.append(self.model.getSolvingStats())


class PolicyBranching(scip.Branchrule):
	'''
	Node of the tree

	Params
	---------
	policy : Dict
		Contains name, type, seed of policy
	mcts_tree : Tree
		Tree object
	'''

	def __init__(self, policy):
		super().__init__()

		self.policy_type = policy['type']
		self.policy_name = policy['name']
		if self.policy_type == 'internal':
			self.policy = policy['name']

		# For internal branching
		self.solving_stats = []

	def branchinitsol(self):
		self.ncutoffs = 0
		self.state_buffer = {}
		self.khalil_root_buffer = {}

	def add_tree(self, mcts_tree):
		# When tree is initialized, add it to brancher
		self.mcts_tree = mcts_tree

	def find_obj_by_attr(self, obj_list, attr):
		''' Finds the object which name matches in object list'''
		obj = None
		for x in obj_list:
			if x.name == attr:
				obj = x
				break
		return obj 

	def find_node_by_name_and_dir(self, obj_list, attr, direction):
		''' Finds the object which name matches in object list'''
		obj = None
		for x in obj_list:
			if x.scip_name == attr:
				if x.branch_up == direction:
					obj = x
					break
		return obj 

	def find_closest_least_fract(self, list_, i):
		min_val = 1.
		min_idx = 0
		for v, idx in enumerate(list_):
			if min(v,1-v) < min_val:
				min_val = v
				min_idx = idx
		return min_idx, min_val


	def branchexeclp(self, allowaddcons):

		if self.policy_type == 'internal':
			result = self.model.executeBranchRule(self.policy, allowaddcons)
		# MCTS policies
		else:
			all_vars = self.model.getVars()
			candidate_vars = self.model.getLPBranchCands()[0]
			khalil_state = self.model.getKhalilState({}, candidate_vars)

			if self.mcts_tree.created_node and self.mcts_tree.curr_node.parent is not None:
				self.mcts_tree.curr_node.set_cands([x.name for x in candidate_vars])

				self.mcts_tree.curr_node.set_ucb_stats()

				self.mcts_tree.created_node = False

			if not self.mcts_tree.scip_has_branched:
				self.mcts_tree.scip_has_branched = True

			if self.mcts_tree.phase == 'add_root_node':
				self.mcts_tree.root_lp_sol = self.model.getObjVal()

				node = Node(None)
				node.set_cands([x.name for x in candidate_vars])
				node.set_ucb_stats()

				print(f"CANDIDATE VARS AT ROOT\n{len(candidate_vars)} FRACT VARS AT ROOT \n{candidate_vars}")


				self.mcts_tree.curr_node = node
				print("ROOT NODE:", self.mcts_tree.curr_node)

				self.mcts_tree.update_root(node)

				self.mcts_tree.phase = 'rollout'
				result = scip.SCIP_RESULT.REDUCEDDOM

			elif self.mcts_tree.phase == 'rollout':
				# Need to get to leaf before we dive
				# Will only start at false for the first node
				while not self.mcts_tree.curr_node.is_leaf():
					cand_name, direction, node_ind = self.mcts_tree.curr_node.choose_candidate()

					# If solution is already feasible
					if cand_name is None:
						return {'result': scip.SCIP_RESULT.REDUCEDDOM}

					cand_node = self.find_node_by_name_and_dir(mcts_tree.curr_node.children, cand_name, direction)

					# Cand exists
					if cand_node is not None:
						self.mcts_tree.mt_second_depth = True
						cand_scip = self.find_obj_by_attr(all_vars, cand_node.scip_name.replace('t_',''))

						self.mcts_tree.update_focus_node(cand_node, node_ind)

						# Branch scip model
						if direction == 0:
							ub = cand_scip.getUbLocal()
							self.model.tightenVarLb(cand_scip, ub)
						else:
							lb = cand_scip.getLbLocal()	
							self.model.tightenVarUb(cand_scip, lb)
					else:
						# For second or more level of the tree
						break

				cand_name, direction, node_ind = self.mcts_tree.curr_node.choose_candidate()

				# If solution is already feasible
				if cand_name is None:
					return {'result': scip.SCIP_RESULT.REDUCEDDOM}

				c_up = khalil_state['ps_up']
				c_down = khalil_state['ps_down']

				cand_scip = self.find_obj_by_attr(all_vars, cand_name.replace('t_',''))
				
				# Branch scip model
				if direction == 0:
					ub = cand_scip.getUbLocal()
					self.model.tightenVarLb(cand_scip, ub)
				else:
					lb = cand_scip.getLbLocal()	
					self.model.tightenVarUb(cand_scip, lb)


				result = scip.SCIP_RESULT.REDUCEDDOM
					
				# Dive when we get to leaf node
				self.mcts_tree.phase = 'dive'

				c_up = khalil_state['ps_up']
				c_down = khalil_state['ps_down']

				# Creates Node as new leaf of the tree
				new_node = Node(self.mcts_tree.curr_node, cand_name, node_ind, direction, cost_up = c_up, cost_down = c_down)

				self.mcts_tree.curr_node = new_node
				self.mcts_tree.created_node = True


			elif self.mcts_tree.phase == 'dive':
				start_var = self.mcts_tree.start_var

				# Diving
				# Will never be false, however fct will not be called when optimal
				if self.model.getStatus() == 'unknown':
					# Branching n times before solving LP
					for i in range(self.mcts_tree.branches_per_lp_solve):
						# Here is the real diving part, as opposed to the 
						if self.policy_type == 'MCTS_fract_diving':
							# Making sure we have enough candidates
							if len(candidate_vars) >= i+1:
								fract_vals = self.model.getLPBranchCands()[1]
								fract_vals_std = [x if x < 0.5 else 1.-x for x in fract_vals]
								min_val = nsmallest(i+1, fract_vals_std)[-1]
								min_ind = fract_vals_std.index(min_val)
								# Get candidate and direction to branch on
								cand_scip = candidate_vars[min_ind]
								direction = int(fract_vals[min_ind] < 0.5)
							else:break
						elif self.policy_type == 'MCTS_vanilla':
							if len(candidate_vars) >= i+1:
								cand_scip = candidate_vars[i]
								direction = int(random.random() > 0.5)
							else:break
						else:
							if len(candidate_vars) >= i+1:
								cand_scip = candidate_vars[i]
								direction = int(random.random() > 0.5)
							else:break

						lb = cand_scip.getLbLocal()
						ub = cand_scip.getUbLocal()

						if direction == 0:
							self.model.tightenVarUb(cand_scip, lb)
						else:
							self.model.tightenVarLb(cand_scip, ub)

					result = scip.SCIP_RESULT.REDUCEDDOM

				else:
					print("MODEL STATUS NOT UNKNOWN")

			if result == scip.SCIP_RESULT.REDUCEDDOM:
				self.mcts_tree.ndomchgs += 1
				self.mcts_tree.ndomchgs_graph += 1

		return {'result': result}



def tree_search(tree, instance_file, return_dict, time_limit, storing_vals):
	'''
	Performs MCTS with time limit (seconds)
	'''
	global m

	start_time = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M:%S")

	graph_sols = False
	graph_root_ucb = False

	store_stats, store_root_ucb, store_root_wins, store_root_visits, store_every_n, folder_name, filename = storing_vals
			
	# GRAPHING
	plt.ion()
	plt.show()
	if graph_sols:
		plt.xlabel("Nb domain changes")
		plt.ylabel("Feasible solution")
	elif graph_root_ucb:
		plt.xlabel("UCB value")
		plt.ylabel("Root candidate")		

	start = time.time()
	nb_dives = 0

	# Problem with presolving when not freed before starting
	m.freeProb()
	m.readProblem(f"{instance_file}")

	# Stopped with thread timeout
	av_rollout_sols = []
	all_sols = []

	create_cols = True

	create_dfs = True

	while True:
		# Rollout + dive
		m.optimize()

		if create_dfs:
			if not os.path.exists(folder_name):
				os.makedirs(folder_name)
			colnames = tree.get_root().candidates
			create_dfs = False
			if store_stats:
				if store_root_ucb:
					df_root_ucb = pd.DataFrame(columns=colnames)
				if store_root_wins:
					df_root_wins = pd.DataFrame(columns=colnames)
				if store_root_visits:
					df_root_visits = pd.DataFrame(columns=colnames)

		# If already optimized
		if not tree.scip_has_branched:
			print("Problem already optimized")
			print(m.getObjVal())
			break

		tree.rollout_tree_nodes.append(tree.curr_node)

		# Initiate solution
		sol = None

		# If feasible, add sol to list
		if m.getStatus() == 'optimal':
			sol = m.getObjVal()
			all_sols.append(sol)

			tree.rollout_sols[tree.rollout_nb-1] = sol
			if sol < tree.curr_best_sol:
				# Add sol and brancher nb to list for graph
				tree.best_sol_list.append(sol)
				tree.nb_lp_solves.append(tree.ndomchgs)

				tree.curr_best_sol = sol

				tree.update_graph = True
		else:
			tree.increment_to_root(node=tree.curr_node, sol=sol)

			m.freeProb()

			m.readProblem(f"{instance_file}")

			utilities.init_scip_params(m, seed=seed, heuristics=False, presolving=False, separating=False, conflict=False)

			m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
			m.setRealParam('limits/time', time_limit)
			
			tree.phase = 'rollout'

			if tree.created_node:
				print("CREATED NODE PROBLEM")

			tree.curr_node = tree.get_root()

			print("NOT OPTIMAL")
			continue


		# GRAPHING SOL VS DOM CHANGES
		if graph_sols: 
			if tree.ndomchgs_graph > 500 and tree.update_graph:
				plt.plot(tree.nb_lp_solves, tree.best_sol_list, 'red')
				plt.draw()
				plt.pause(0.001)

				tree.update_graph = False

				tree.ndomchgs_graph = 0

		# GRAPHING ROOT CHILD UCB
		elif graph_root_ucb:
			if random.random() < 0.1:
				plt.bar(list(range(len(tree.root_node.child_ucb[:25]))), tree.root_node.child_ucb[:25], color='red')
				plt.draw()
				plt.title(f"{nb_dives} dives")
				plt.ylim(min(tree.root_node.child_ucb),max(tree.root_node.child_ucb))
				plt.pause(0.001)

		# Increment nb of runs
		tree.increment_to_root(node=tree.curr_node, sol=sol)

		tree.phase = 'rollout'

		if nb_dives % store_every_n == 0 : 			
			if store_stats:
				if store_root_ucb:
					df_root_ucb.loc[nb_dives] = pd.Series(tree.get_root_ucb(),index=df_root_ucb.columns)
					utilities.log_stats(df_root_ucb, folder_name, filename ,'ucb',start_time)
				if store_root_wins:
					df_root_wins.loc[nb_dives] = pd.Series(tree.get_root().child_nb_wins,index=df_root_wins.columns)
					utilities.log_stats(df_root_wins,folder_name, filename ,'wins',start_time)
				if store_root_visits:
					df_root_visits.loc[nb_dives] = pd.Series(tree.get_root().child_nb_visits,index=df_root_visits.columns)
					utilities.log_stats(df_root_visits,folder_name, filename ,'visits',start_time)

		nb_dives += 1		

		# Add new nodes
		if tree.rollout_nb == tree.max_rollout_nb:
			tree.rollout_nb = 1

			if tree.adding_new_nodes == 'all':
				tree.add_all_nodes()
			elif tree.adding_new_nodes == 'best':
				tree.add_best_node()
			elif tree.adding_new_nodes == 'random':
				tree.add_rand_node()		

			best_sol = min(tree.rollout_sols)
			av_rollout_sol = sum(tree.rollout_sols)/len(tree.rollout_sols)
			print(f"{tree.rollout_sols}, Best: {tree.get_best_sol()}, Av Sol: {av_rollout_sol}" )
			best_node = tree.rollout_tree_nodes[tree.rollout_sols.tolist().index(best_sol)]
			tree.increment_to_root(best_node, wins=True, sol=best_sol)

			# Reset lists
			tree.rollout_sols = np.ones(tree.max_rollout_nb) * np.inf
			tree.rollout_tree_nodes = []

			time_elapsed = (time.time() - start)

			print(f"AV TIME PER DIVE: {round((time_elapsed/nb_dives),2)}, {nb_dives} dives, {utilities.get_mins_left(time_elapsed, time_limit)} left\n")

			tree.start_var = False
			av_rollout_sols.append(av_rollout_sol)

		else:
			# Only for the first n rollouts
			if nb_dives < tree.max_rollout_nb:
				tree.start_var = True
			tree.rollout_nb += 1

		

		faulthandler.enable()

		# Restart problem
		m.freeProb()

		m.readProblem(f"{instance_file}")

		utilities.init_scip_params(m, seed=seed, heuristics=False, presolving=False, separating=False, conflict=False)

		m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
		m.setRealParam('limits/time', time_limit)

		tree.curr_node = tree.get_root()

		return_dict['feas_sols_graph'], return_dict['nb_lp_solves_graph'] = tree.get_graph_vals()
		return_dict['root_child_ucb'] = tree.get_root_ucb()
		return_dict['root_nb_wins'], return_dict['root_nb_visits'] = tree.get_root_stats()
		return_dict['all_feas_sols'] = all_sols


if __name__ == '__main__':
	instance_file = sys.argv[1]
	seed = 1
	time_limit = 3600. * 2
	time_limit = 30
	episode = 1

	#################################
	# HYPER PARAMETERS OF THE MODEL #
	#################################
	EXP_RATIO = np.sqrt(2)
	BRANCHES_PER_LP_SOLVE = 5

	#################################
	#        STORING OF MODEL       #
	#################################
	store_stats = True
	store_root_ucb = True
	store_root_wins = True
	store_root_visits = True
	store_every_n = 100
	filename = 'sqrt_2_exp_ratio'
	folder_name = 'data_test'
	storing_vals = (store_stats, store_root_ucb, store_root_wins, store_root_visits, store_every_n, folder_name, filename)


	instances = [{'path':instance_file,'type':'setcover'}]

	branching_policies = [{'type':'internal','name':'internal','seed':0, 'dive_type':None},
						  {'type':'MCTS_vanilla','name':'MCTS_vanilla','seed':0, 'dive_type':'random'},
						  {'type':'MCTS_hot_start','name':'MCTS_hot_start','seed':0, 'dive_type':'random'},
						  {'type':'MCTS_coef_diving','name':'MCTS_coef_diving','seed':0, 'dive_type':'coef'},
						  {'type':'MCTS_fract_diving','name':'MCTS_fract_diving','seed':0, 'dive_type':'fract'}]

	branching_policies = [{'type':'MCTS_vanilla','name':'MCTS_vanilla','seed':0}]

	# Dict {method_name:[list_of_primal_integral]}
	# agg_integral_stats = {'internal':[],'MCTS_vanilla':[], 'MCTS_hot_start':[]}
	agg_integral_stats = {x['type']:[] for x in branching_policies}

	# Same, but each containing the raw info (nb lp solves and feas solutions) 
	raw_stats = {'internal':[],'MCTS_coef_diving':[], 'MCTS_coef_diving':[]}
	for instance in instances:
		for policy in branching_policies:
			brancher = PolicyBranching(policy)

			# Model will be initialized
			m = scip.Model()
			m.setIntParam('display/verblevel', 0)

			m.readProblem(f"{instance['path']}")

			m.includeEventhdlr(SolvingStatsRecorder(brancher.solving_stats), "SolvingStatsRecorder", "")

			# AGGRESSIVE MODE FOR INTERNAL BRANCHING
			if policy['type'] == 'internal':
				utilities.init_scip_params(m, seed=seed, heuristics='agg', presolving=False, separating=False, conflict=False)
			else:
				utilities.init_scip_params(m, seed=seed, heuristics=False, presolving=False, separating=False, conflict=False)

			# if policy['type'] == 'MCTS_coef_diving':
			# 	utilities.disable_all_but_coef(m)
			# elif policy['type'] == 'MCTS_fract_diving':
			# 	utilities.disable_all_but_fract(m)

			m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
			m.setRealParam('limits/time', time_limit)

			if policy['type'] == 'internal':
				m.optimize()

				nlps = [x['nlps'] for x in brancher.solving_stats]
				primalbounds = [x['primalbound'] for x in brancher.solving_stats]
				raw_stats['internal'].append((nlps, primalbounds))
				agg_integral_stats['internal'].append(primal_integral(nlps, primalbounds))
			else:
				# MCTS BASED METHOD
				mcts_tree = Tree(policy['seed'], brancher, branches_per_lp_solve=BRANCHES_PER_LP_SOLVE, exp_ratio=EXP_RATIO)
				
				brancher.add_tree(mcts_tree)

				m.includeBranchrule(
					branchrule=brancher,
					name=f"{policy['type']}:{policy['name']}",
					desc=f"Custom PySCIPOpt branching policy.",
					priority=666666, maxdepth=-1, maxbounddist=1)

				sys.setrecursionlimit(2097152)

				manager = Manager()
				return_dict = manager.dict()

				# Start 
				p = Process(target=tree_search, args=(mcts_tree, instance['path'], return_dict, time_limit, storing_vals))
				p.start()
				p.join(timeout=time_limit)

				ma_nb = 50
				ma_vals = utilities.moving_average(return_dict['all_feas_sols'], ma_nb)
				x_axis_ma = [x for x in range(len(return_dict['all_feas_sols'])) if x>(ma_nb-2)]


				print(return_dict)

				if p.is_alive():
					p.terminate()

				plt.plot(range(len(return_dict['all_feas_sols'])),return_dict['all_feas_sols'])
				plt.plot(x_axis_ma, ma_vals)
				plt.show()

				raw_stats[policy['type']].append((return_dict['nb_lp_solves_graph'], return_dict['feas_sols_graph']))
				agg_integral_stats[policy['type']].append(primal_integral(return_dict['nb_lp_solves_graph'], return_dict['feas_sols_graph']))

			m.freeProb()

			print(return_dict)


	for i in range(10):print("")
	print(raw_stats)
	print(agg_integral_stats)













