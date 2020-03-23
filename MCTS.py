import pyscipopt as scip
import utilities
import math
import multiprocessing as mp
import sys
import numpy as np

class Tree:
	''' 
	Simple tree which will be used MCST


	Parameters
	-------------
	root_params : starting state of the relaxed LP problem


	'''

	def __init__(self, root_params):
		''' 
		Only starts with root node.
		'''
		self.nodes = [Node(None,root_params)]


	def add_candidates(self):
		''' 
		Adds candidates for a certain split
		Candidates are all of the non integer variables which can be branched on
		'''
		pass


	def increment_to_root(self, node):
		'''
		Once we arrive at the end of the episode, we increment path back to root
		'''
		while node.antecedant is not None:
			# Extract validity of solution from state
			node.node_stats += node.evaluate_solution() 
			increment_to_root(node.antecedant)


	def choose_candidate(self, node, exp_ratio=math.sqrt(2)):
		'''
		Candidate is chosen based on balance between number of visits and "wins"
		'''
		scores = np.array()
		for cand in node.candidates:
			# Wins at current node
			wins = cand.node_stats[1]
			# Nb of simultions which passed through current node
			n = sum(cand.node_stats)
			cap_N = sum(cand.antecedant.node_stats)

			# Upper confidence bound
			score = (wins-n) + exp_ratio*math.sqrt(math.log(cap_N)/n)
			scores.append(score)

		# Get max scored value node
		return nodes.candidates[np.argmax(scores)]






class Node:
	'''
	Node of the tree

	Params
	---------
	antecendant : Node
		Antecedant of current node, with None if root node
	lp_sol : float
		Repaxed problem solution at current node
	curr_state : dict
		Current state of the LP at the current node
	is_leaf_node : bool
		node is terminal
	valid_sol : bool
		solution is valid (integer)
	nb_outcomes : int
		Number of possible outcomes for the MCST. >=2, one for valid and one for not valid
	'''

	def __init__(self, antecendant, lp_sol, curr_state, nb_outcomes=2, is_leaf_node=False, valid_sol=False):
		self.antecendant = antecendant
		self.lp_sol = lp_sol
		self.curr_state = curr_state
		self.is_leaf_node = is_leaf_node
		self.valid_sol = valid_sol
		self.nb_outcomes = nb_outcomes
		# Node stats is a list containing number of each of the possible outcomes (valid, not valid, etc) 
		self.node_stats = []
		# If candidate list is empty, we are at root node
		self.candidates = []


	def get_antecendant(self):
		return self.antecendant 


	def evaluate_solution(self):
		''' 
		Returns 0 if solution is not feasible, 1 if feasible
		'''
		return a


	def get_and_create_candidates(self, cur_node):
		'''
		Gather list of candidates based on current node and create object
		'''
		if not is_leaf_node:
			for candidate in candidates:
				cand_sol = None
				cand_state = None
				valid_sol = None

				# Create node with current node as antecendant
				n = Node(cur_node, cur_sol, cur_state, valid_sol)
				self.candidates.append(n)


if __name__ == '__main__':
	instance = sys.argv[1]
	seed = 1
	time_limit = 3600
	episode = 1

	m = scip.Model()
	# print(dir(m))
	# Disable heuristics
	# m.setHeuristics(scip.SCIP_PARAMSETTING.OFF)

	print(f'{instance}')
	m.setIntParam('display/verblevel', 0)
	m.readProblem(f'{instance}')

	# utilities.init_scip_params(m, seed=seed)
	m.setIntParam('timing/clocktype', 2)

	m.setRealParam('limits/time', time_limit) 


	m.createSol()
	# print(m.getLPSolstat())
	print(m.getVars())

	cands, *_ = m.getPseudoBranchCands()








