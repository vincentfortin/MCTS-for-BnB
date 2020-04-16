import pyscipopt as scip
import utilities
import math
import multiprocessing as mp
import sys
import numpy as np
import time
import re
import random

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

class PolicyBranching(scip.Branchrule):
	
	def __init__(self, policy):
		super().__init__()

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

		print(self.model.getObjVal())
		# print(candidate_vars)
		print(len(candidate_vars))
		cand = candidate_vars[0]

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

	branching_policies = [{'type':'testMCTS','name':'MCTS','seed':0}]

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


	for instance in instances:
		print(f"{instance['type']}: {instance['path']}...")

		for policy in branching_policies:
			m = scip.Model()
			m.setIntParam('display/verblevel', 0)
			m.readProblem(f"{instance['path']}")
			utilities.init_scip_params(m, seed=seed, heuristics=False, presolving=False, separating=False, conflict=False)
			m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
			m.setRealParam('limits/time', time_limit)

			brancher = PolicyBranching(policy)
			m.includeBranchrule(
			    branchrule=brancher,
			    name=f"{policy['type']}:{policy['name']}",
			    desc=f"Custom PySCIPOpt branching policy.",
			    priority=666666, maxdepth=-1, maxbounddist=1)

			walltime = time.perf_counter()
			proctime = time.process_time()


			m.optimize()

			print(m.getStatus())
			print(m.getObjVal())
			print(m.getDualbound())
			print(m.getPrimalbound())

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








