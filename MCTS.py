import pyscipopt as scip
import utilities
import math
import multiprocessing as mp
import sys
import numpy as np
import time
import re
import random


class PolicyBranching(scip.Branchrule):
	
	def __init__(self, policy, mcts_tree):
		super().__init__()

		self.mcts_tree = mcts_tree

		self.policy_type = policy['type']
		self.policy_name = policy['name']
		if self.policy_type == 'internal':
			self.policy = policy['name']

	def branchinitsol(self):
		self.ndomchgs = 0
		self.ncutoffs = 0
		self.state_buffer = {}
		self.khalil_root_buffer = {}


	def branchexeclp(self, allowaddcons):
		candidate_vars, *_ = self.model.getPseudoBranchCands()

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


		elif self.mcts_tree.phase == 'dive':

			start_var = True
			# Diving
			# Will never be false, however fct will not be called when optimal
			if self.model.getStatus() == 'unknown':
				if start_var:
					# First 
					cand, direction, node_ind = self.mcts_tree.curr_node.choose_candidate()
					self.mcts_tree.curr_node.set_cands(candidate_vars)
					self.mcts_tree.curr_node.update_best_sol(self.model.getObjVal())
					cand, direction, node_ind = self.mcts_tree.curr_node.choose_candidate()
					# Add branched var to current node
					# self.mcts_tree.curr_node.add_branched_var(cand)
					self.mcts_tree.rollout_parent_index.append(node_ind)

					start_var = False
				else:
					cand = candidate_vars[0]
					
				lb = cand.getLbLocal()
				ub = cand.getUbLocal()


				if random.random() > 0.5:
					# Change upper or lower bound so that ub == lb
					self.model.tightenVarLb(cand, ub)
				else:
					self.model.tightenVarUb(cand, lb)

				result = scip.SCIP_RESULT.REDUCEDDOM

				self.mcts_tree.rollout_nb += 1


		return {'result': result}




class Tree():
	''' 
	Simple tree which will be used MCST


	Parameters
	-------------
	problem_name : String
		Name of the linear program to be solved

	adding_new_nodes : String (all, best, random)
		How should new nodes be added
	'''

	def __init__(self, problem_name, brancher, adding_new_nodes='all'):
		''' 
		Only starts with root node.
		'''
		super().__init__()

		self.adding_new_nodes = adding_new_nodes

		# Model will be initialized
		self.m = scip.Model()
		self.m.setIntParam('display/verblevel', 0)
		self.m.readProblem(f"{problem_name}")

		utilities.init_scip_params(self.m, seed=seed, heuristics=False, presolving=False, separating=False, conflict=False)

		self.m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
		self.m.setRealParam('limits/time', time_limit)

		self.m.includeBranchrule(
			branchrule=brancher,
			name=f"{policy['type']}:{policy['name']}",
			desc=f"Custom PySCIPOpt branching policy.",
			priority=666666, maxdepth=-1, maxbounddist=1)

		# For root node
		root_lp_sol = self.m.getObjVal()
		root_cands, *_  = self.getPseudoBranchCands()

		self.nodes = [Node(None, root_cands, root_lp_sol)]

		# Focus node
		self.curr_node = nodes[0]

		# Phase in ['rollout', 'add_node', 'dive']
		# Starts at dive and since we only have root node and no rollout is needed
		self.phase = 'dive'

		# Rollout stats
		self.rollout_nb = 1
		self.max_rollout_nb = 4

		self.rollout_sols = np.inf(max_rollout_nb)
		self.rollout_scip_nodes = []
		# Those are the parent of the nodes which hae a dive performed on, since those nodes may not be created
		self.rollout_tree_nodes = []
		self.rollout_parent_index = []



	def tree_search(self, limit = 300):
		'''
		Performs MCTS with time limit (seconds)
		'''
		for i in range(100):
			# Rollout + dive
			self.m.optimize()

			self.rollout_tree_nodes.append(self.curr_node)
			self.rollout_scip_nodes.append(self.curr_node.scip_node)

			# I feasible, add sol to list
			if self.m.getStatus() == 'optimal':
				self.rollout_sols[rollout_nb-1] = self.m.getObjVal()

			# Increment nb of runs
			self.increment_to_root(node=self.curr_node)

			# Add new nodes
			if self.rollout_nb == self.max_rollout_nb:
				self.rollout_nb = 1
				self.add_all_nodes()
				
				best_sol = max(self.rollout_sols)
				best_node = self.rollout_tree_nodes[rollout_sols.index(best_sol)]
				self.increment_to_root(wins=True)

				# Reset lists
				self.rollout_sols = np.inf(max_rollout_nb)
				self.rollout_scip_nodes = []
				self.rollout_tree_nodes = []
				self.rollout_parent_index = []

				if self.adding_new_nodes == 'all':
					self.add_all_nodes()
				elif self.adding_new_nodes == 'best':
					self.add_best_node()
				elif self.adding_new_nodes == 'random':
					self.add_rand_node()





	def reset_model(self):
		''' Resets model after dive'''
		self.m.freeTransform()


	def branchinitsol(self):
		self.ndomchgs = 0
		self.ncutoffs = 0
		self.state_buffer = {}
		self.khalil_root_buffer = {}


	def get_root(self):
		'''
		Returns root node
		'''
		return self.nodes[0]


	def add_all_nodes(self):
		''' 
		Adds a new node for each node which had a dive performed on
		'''
		for parent, scip_node, node_ind in zip(self.rollout_tree_nodes, self.rollout_scip_nodes, self.rollout_parent_index):
			self.add_node(parent, scip_node, node_idx, node_idx%2)



	def add_best_node(self):
		''' 
		Adds a new node at the node where best dive was
		'''
		max_val = max(self.rollout_sols)
		max_idx = self.rollout_sols.index(max_val)
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


	def add_node(self, parent, scip_node, node_idx, branch_up):
		'''
		Adds node as a new leaf of the tree.
		'''
		child = Node(parent, scip_node, ind_of_parent=node_idx, branch_up=branch_up)

		# Parent is no longer leaf when all nodes have been visited
		if sum(parent.child_ucb == np.infinity) == 0:
			parent.is_leaf_node = False

		parent.add_children(child)

		self.nodes.append(child)



	def update_focus_node(self, cand, node_ind):
		''' 
		Updates current node in tree, as well as the index of it's parent
		'''
		self.curr_node = cand
		self.curr_node.ind_of_parent = node_ind


	def increment_to_root(self, node=None, wins=False):
		'''
		Once we arrive at the end of the episode, we increment path back to root
		We first need to find the best node
		If all infeasible or all same sol, we choose randomly one of the sols
		node : Node to increment stats on. If wins, we don't have a node, since we will chose from list of rollout nodes
		wins : If true, we increment number of wins at node, otherwise, we increment nb runs
		'''
		parent_node = node.parent
		node_index = node.ind_of_parent
		# Extract validity of solution from state
		if wins:
			parent_node.child_nb_wins[node_index] += 1
			parent_node.nb_wins += 1
		else:
			parent_node.child_nb_visits[node_index] += 1
			parent_node.nb_visits += 1

		parent_node.update_ucb()

		# Only increment if we are not at root
		if node.parent is not None:
			increment_to_root(parent_node, wins)



class Node():
	'''
	Node of the tree

	Params
	---------
	parent : Node
		parent of current node, with None root node
	lp_sol : float
		Repaxed problem solution at current node
	candidates: list
		List of scip nodes candidates
	ind_of_parent: int
		Index of parent's candidates. None if no parent node (root)
	scip_node: Scip node
		Scip node associated with MCTS node. 
		None when at root
	branch_up: Int or None
		Wether node is branched up (0) or down (1)
		None when at root
	'''

	def __init__(self, parent, scip_node = None, ind_of_parent=None, branch_up=None):

		self.parent = parent
		self.curr_state = curr_state

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


	def add_children(self, child):
		'''
		Adds children to current node
		child is a Node
		'''
		self.children.append(child)


	def get_stats(self):
		'''
		Returns node UCB stats in the form of a tuple (runs, wins)
		'''
		return (self.nb_visits, self.nb_wins)


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
		self.child_ucb = np.inf(len(self.child_nb_wins))



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
		max_ind = self.child_ucb.index(max_val)
		cand = self.candidates[max_ind]
		direction = max_ind % 2

		return cand, direction, max_ind


	def choose_children(self):
		'''
		Choose children based on UCT score
		Returns both the children and the direction of branching
		'''
		cand_ind = np.argmax([cand.ucb_score for cand in self.children])
		cand = self.children[cand_ind]
		direction = cand.branch_up
		return cand, direction


	def set_cands(self, cands):
		''' 
		Sets a list of candidates for node
		'''
		# Duplicates each element (first is up, second is down)
		self.candidates = [x for x in cands for _ in (0,1)]
		self.cand_visits = np.zeros(len(self.candidates))
		self.cand_wins = np.zeros(len(self.candidates))


	def set_lp_sol(self, lp_sol):
		'''
		Sets LP obj function value at current node
		'''
		self.lp_sol = lp_sol


	def is_leaf(self):
		''' 
		Returns true if node is node is leaf node
		'''
		return self.leaf_node




class PolicyBranching2(scip.Branchrule):
	
	def __init__(self, policy):
		super().__init__()

		self.counter = 0

		self.policy_type = policy['type']
		self.policy_name = policy['name']
		if self.policy_type == 'internal':
			self.policy = policy['name']

	def branchinitsol(self):
		self.ndomchgs = 0
		self.ncutoffs = 0
		self.state_buffer = {}
		self.khalil_root_buffer = {}

	def branchexeclp(self, allowaddcons):
		candidate_vars, *_ = self.model.getPseudoBranchCands()
		candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

		# print(candidate_vars)
		# print("________________")
		# print(len(candidate_vars))
		cand = candidate_vars[0]

		# print(dir(self.model))
		# print(a)


		# self.model.startDive()
		# self.model.solveDiveLP()

		# print(self.model.getStatus())
		# print(self.model.getObjVal())

		# if model.getStatus() == SCIP_STATUS_OPTIMAL:
		# 	self.rollout_sols[rollout_nb] = model.getObjVal()
		# self.rollout_nodes.append(self.curr_node)

		# self.increment_to_root(curr_node)

		# if self.rollout_nb == self.max_rollout_nb:
		# 	# Increment wins, add nodes
		# 	self.increment_to_root(current_node)

		# 	self.rollout_nb = 1
		# else:
		# 	self.rollout_nb += 1

		# self.model.freeTransform()

		lb = cand.getLbLocal()
		ub = cand.getUbLocal()


		if random.random() > 0.5:
			# Change upper or lower bound so that ub == lb
			self.model.tightenVarLb(cand, ub)
		else:
			self.model.tightenVarUb(cand, lb)


		result = scip.SCIP_RESULT.REDUCEDDOM

        # fair node counting
		if result == scip.SCIP_RESULT.REDUCEDDOM:
			self.ndomchgs += 1
		elif result == scip.SCIP_RESULT.CUTOFF:
			self.ncutoffs += 1

		return {'result': result}



if __name__ == '__main__':
	instance_file = sys.argv[1]
	seed = 1
	time_limit = 3600
	episode = 1

	# result_file = f"{args.problem}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
	instances = [{'path':instance_file,'type':'setcover'}]

	branching_policies = [{'type':'MCTS','name':'MCTS','seed':0}]

	fieldnames = [
	    'policy',
	    'seed',
	    'type',
	    'instance',
	    'nnodes',
	    'nlps',
	    'stime',
	    'gap',
	    'status',
	    'ndomchgs',
	    'ncutoffs',
	    'walltime',
	    'proctime',
	]


	mcts_tree = Tree(instances[0]['path'], branching_policies[0])
	brancher = PolicyBranching(policy, mcts_tree)

	mcts_tree.tree_search(limit=300)


			for i in range(4):
				walltime = time.perf_counter()
				proctime = time.process_time()

				m.optimize()

				brancher.counter = i

				print(m.getStatus())
				print(m.getObjVal())
				print(m.getDualbound())
				print(m.getPrimalbound())

				m.freeTransform()

			print(brancher.counter)

			walltime = time.perf_counter() - walltime
			proctime = time.process_time() - proctime

			stime = m.getSolvingTime()
			nnodes = m.getNNodes()
			nlps = m.getNLPs()
			gap = m.getGap()
			status = m.getStatus()
			ndomchgs = brancher.ndomchgs
			ncutoffs = brancher.ncutoffs

			# writer.writerow({
   #              'policy': f"{policy['type']}:{policy['name']}",
   #              'seed': policy['seed'],
   #              'type': instance['type'],
   #              'instance': instance,
   #              'nnodes': nnodes,
   #              'nlps': nlps,
   #              'stime': stime,
   #              'gap': gap,
   #              'status': status,
   #              'ndomchgs': ndomchgs,
   #              'ncutoffs': ncutoffs,
   #              'walltime': walltime,
   #              'proctime': proctime,
			# })

			# csvfile.flush()
			m.freeProb()

			print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} ({nnodes+2*(ndomchgs+ncutoffs)}) nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")















