import pyscipopt as scip
import utilities
import math
import multiprocessing as mp
import sys
import numpy as np
import time
import re
import random
from multiprocessing import Process, Value, Manager
import faulthandler
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from scipy import integrate

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

		self.nodes = []

		# Focus node
		self.curr_node = None

		# Phase in ['rollout', 'add_node', 'dive']
		# Starts at dive and since we only have root node and no rollout is needed
		self.phase = 'add_root_node'

		self.adding_new_nodes = adding_new_nodes

		self.scip_has_branched = False

		self.start_var = True

		# Rollout stats
		self.rollout_nb = 1
		self.max_rollout_nb = 4

		self.rollout_sols = np.ones(self.max_rollout_nb) * np.inf
		self.rollout_scip_nodes = []
		# Those are the parent of the nodes which hae a dive performed on, since those nodes may not be created
		self.rollout_tree_nodes = []
		self.rollout_parent_index = []
		self.rollout_node_cands = []

		self.best_sol_list = []
		self.nb_lp_solves = []

		self.curr_best_sol = 1e8

		self.ndomchgs = 0
		self.ndomchgs_graph = 0

		self.update_graph = True

		self.exp_ratio = exp_ratio


	def get_best_sol(self):
		''' 
		Returns best solution from tree
		'''
		return self.nodes[0].best_lp_sol


	def add_all_nodes(self):
		''' 
		Adds a new node for each node which had a dive performed on
		'''
		for parent, scip_node, node_idx, node_cands in zip(self.rollout_tree_nodes, self.rollout_scip_nodes, self.rollout_parent_index, self.rollout_node_cands):
			self.add_node(parent, scip_node, node_idx, node_idx%2, node_cands)

	def add_best_node(self):
		''' 
		Adds a new node at the node where best dive was
		'''
		max_val = min(self.rollout_sols)
		max_idx = self.rollout_sols.tolist().index(max_val)
		node = self.rollout_tree_nodes[max_idx]
		scip_node = self.rollout_scip_nodes[max_idx]
		node_idx = self.rollout_parent_index[max_idx]

		self.add_node(node, scip_node, node_idx, node_idx%2)
		

	def add_rand_node(self):
		''' 
		Adds a new node randomly between the nodes which had dive performed on
		'''
		rand_idx = random.randint(0, self.max_rollout_nb-1)
		node = self.rollout_tree_nodes[rand_idx]
		scip_node = self.rollout_scip_nodes[rand_idx]
		node_idx = self.rollout_parent_index[rand_idx]

		self.add_node(node, scip_node, node_idx, node_idx%2)		


	def add_node(self, parent, scip_node, node_idx, branch_up, cands):
		'''
		Adds node as a new leaf of the tree.
		'''
		child = Node(parent, scip_node, ind_of_parent=node_idx, branch_up=branch_up)

		# Parent is no longer leaf when all nodes have been visited
		if len(parent.child_ucb) == len(parent.children):
			parent.is_leaf_node = False

		child.set_cands(cands)
		child.set_ucb_stats()

		parent.children.append(child)

		# self.nodes.append(child)


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
			increment_to_root(parent_node, wins, sol)
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
		return self.nodes[0].child_ucb


	def get_best_dive_sol(self):
		'''
		Gets the best sol found in the first n dives
		Returns 1e8 otherwise
		'''
		return min(self.rollout_sols)



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
	'''

	def __init__(self, parent, scip_node = None, ind_of_parent=None, branch_up=None):

		self.parent = parent

		self.candidates = None
		# List of all nb_visits for candidates and wins
		self.cand_visits = None
		self.cand_wins = None

		self.scip_node = scip_node

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


	def get_parent(self):
		'''
		Returns a node's parent
		'''
		return self.parent


	def get_best_sol(self):
		'''
		Gets best feasible solution
		'''
		return nodes[0].best_lp_sol


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


	def set_ucb_stats(self):
		'''
		Initialize nb visit, nb wins, ucb stats.
		Not in __init__ function since we don't have candidates at the start
		'''
		# Child stats
		self.child_nb_visits = np.ones(len(self.candidates)) * 1e-8
		self.child_nb_wins = np.zeros(len(self.candidates))

		# Initializes to all infinity
		self.child_ucb = np.ones(len(self.child_nb_wins)) * np.inf



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

	def branchexeclp(self, allowaddcons):
        # SCIP internal branching rule
		if self.policy_type == 'internal':
			result = self.model.executeBranchRule(self.policy, allowaddcons)
		# custom policy branching
		else:
			candidate_vars, *_ = self.model.getPseudoBranchCands()

			if not self.mcts_tree.scip_has_branched:
				self.mcts_tree.scip_has_branched = True

			if self.mcts_tree.phase == 'add_root_node':
				self.mcts_tree.root_cands = candidate_vars
				self.mcts_tree.root_lp_sol = self.model.getObjVal()

				node = Node(None)
				node.set_cands([x.name for x in candidate_vars])
				node.set_ucb_stats()

				self.mcts_tree.curr_node = node

				self.mcts_tree.nodes.append(node)

				self.mcts_tree.phase = 'dive'


			if self.mcts_tree.phase == 'rollout':
				# Need to get to leaf before we dive
				# Here leaf means the node does not have any dives performed on it
				while not self.mcts_tree.curr_node.is_leaf():
					cand, direction, node_ind = self.mcts_tree.curr_node.choose_candidate()
					self.mcts_tree.update_focus_node(cand, node_ind)

					# Branch scip model
					if direction == 0:
						ub = cand.getUbLocal()
						self.model.tightenVarLb(cand, ub)
					else:
						lb = cand.getLbLocal()	
						self.model.tightenVarUb(cand, lb)

				result = scip.SCIP_RESULT.REDUCEDDOM
					
				# Dive when we get to leaf node
				self.mcts_tree.phase = 'dive'
				self.mcts_tree.start_var = True					


			elif self.mcts_tree.phase == 'dive':

				start_var = self.mcts_tree.start_var

				# Diving
				# Will never be false, however fct will not be called when optimal
				if self.model.getStatus() == 'unknown':
					# Branching n times before solving LP
					for i in range(self.mcts_tree.branches_per_lp_solve):
						if start_var:
							cand, direction, node_ind = self.mcts_tree.curr_node.choose_candidate()

							cand = self.find_obj_by_attr(candidate_vars, cand)

							if cand is None:
								print("CAND NOT FOUND IN CAND VARS")

							self.mcts_tree.rollout_parent_index.append(node_ind)
							self.mcts_tree.rollout_node_cands.append([x.name for x in candidate_vars])
							self.mcts_tree.curr_node = Node(self.mcts_tree.curr_node, cand, node_ind, direction)
						else:
							cand = candidate_vars[i]
						
						lb = cand.getLbLocal()
						ub = cand.getUbLocal()

						if start_var:
							# Branch up
							if direction == 0:
								self.model.tightenVarUb(cand, lb)
							else:
								self.model.tightenVarLb(cand, ub)
							self.mcts_tree.start_var = False
							start_var = False
						else:
							if random.random() > 0.5:
								# Change upper or lower bound so that ub == lb
								self.model.tightenVarLb(cand, ub)
							else:
								self.model.tightenVarUb(cand, lb)

					result = scip.SCIP_RESULT.REDUCEDDOM

				else:
					print("MODEL STATUS NOT UNKNOWN")

			if result == scip.SCIP_RESULT.REDUCEDDOM:
				self.mcts_tree.ndomchgs += 1
				self.mcts_tree.ndomchgs_graph += 1

		return {'result': result}


def primal_integral(nlps, primalbounds, zero_val=0):
	'''
	Calculates the integral of the primal bound, over the number of lp solves
	zero_val : Should use the value of the optimal solution. If not found, use zero ?
	'''
	return integrate.simps(np.array(primalbounds)-zero_val, np.array(nlps))
	


def tree_search(tree, instance_file, return_dict):
	'''
	Performs MCTS with time limit (seconds)
	'''

	global m

	graph_sols = False
	graph_root_ucb = False

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

	# Stopped with thread timeout
	while True:
		# Rollout + dive
		m.optimize()

		# If already optimized
		if not tree.scip_has_branched:
			print("Problem already optimized")
			print(m.getObjVal())
			break

		tree.rollout_tree_nodes.append(tree.curr_node)
		tree.rollout_scip_nodes.append(tree.curr_node.scip_node)

		# Initiate solution
		sol = None

		# I feasible, add sol to list
		if m.getStatus() == 'optimal':
			sol = m.getObjVal()

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

			tree.curr_node = tree.nodes[0]

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
			if random.random() < 0.01:
				plt.bar(list(range(len(tree.nodes[0].child_ucb[:25]))), tree.nodes[0].child_ucb[:25], color='red')
				plt.draw()
				plt.title(f"{nb_dives} dives")
				plt.ylim(min(tree.nodes[0].child_ucb),max(tree.nodes[0].child_ucb))
				plt.pause(0.001)

		# Increment nb of runs
		tree.increment_to_root(node=tree.curr_node, sol=sol)

		tree.phase = 'rollout'

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
			print(tree.rollout_sols, tree.get_best_sol())
			best_node = tree.rollout_tree_nodes[tree.rollout_sols.tolist().index(best_sol)]
			tree.increment_to_root(best_node, wins=True, sol=best_sol)

			# Reset lists
			tree.rollout_sols = np.ones(tree.max_rollout_nb) * np.inf
			tree.rollout_scip_nodes = []
			tree.rollout_tree_nodes = []
			tree.rollout_parent_index = []

			nb_dives += 4

			print(f"AV TIME PER DIVE: {(time.time() - start)/nb_dives}, {nb_dives} dives\n")

		else:
			tree.rollout_nb += 1

		faulthandler.enable()

		# Restart problem
		m.freeProb()

		m.readProblem(f"{instance_file}")

		utilities.init_scip_params(m, seed=seed, heuristics=False, presolving=False, separating=False, conflict=False)

		m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
		m.setRealParam('limits/time', time_limit)

		tree.curr_node = tree.nodes[0]

		return_dict['feas_sols_graph'], return_dict['nb_lp_solves_graph'] = tree.get_graph_vals()
		return_dict['root_child_ucb'] = tree.get_root_ucb()




if __name__ == '__main__':
	instance_file = sys.argv[1]
	seed = 1
	time_limit = 30
	episode = 1
	exp_ratio = np.sqrt(2)

	instances = [{'path':instance_file,'type':'setcover'}]

	branching_policies = [{'type':'internal','name':'internal','seed':0},
						  {'type':'MCTS_vanilla','name':'MCTS_vanilla','seed':0},
						  {'type':'MCTS_hot_start','name':'MCTS_vanilla','seed':0}]

	# Dict {method_name:[list_of_primal_integral]}
	agg_integral_stats = {'internal':[],'MCTS_vanilla':[]}
	# Same, but each containing the raw info (nb lp solves and feas solutions) 
	raw_stats = {'internal':[],'MCTS_vanilla':[]}
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

			m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
			m.setRealParam('limits/time', time_limit)

			if policy['type'] == 'internal':
				m.optimize()
				# print(f"OBJ VAL:{m.getObjVal()}")
				nlps = [x['nlps'] for x in brancher.solving_stats]
				primalbounds = [x['primalbound'] for x in brancher.solving_stats]
				raw_stats['internal'].append((nlps, primalbounds))
				agg_integral_stats['internal'].append(primal_integral(nlps, primalbounds))
			else:
				# MCTS BASED METHOD
				mcts_tree = Tree(policy['seed'], brancher, branches_per_lp_solve=4, exp_ratio=exp_ratio)
				
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
				p = Process(target=tree_search, args=(mcts_tree, instance['path'], return_dict))
				p.start()
				p.join(timeout=time_limit)

				if p.is_alive():
					p.terminate()

				raw_stats[policy['type']].append((return_dict['nb_lp_solves_graph'], return_dict['feas_sols_graph']))
				agg_integral_stats[policy['type']].append(primal_integral(return_dict['nb_lp_solves_graph'], return_dict['feas_sols_graph']))

			m.freeProb()

	for i in range(10):print("")
	print(raw_stats)
	print(agg_integral_stats)













