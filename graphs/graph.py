import numpy as np

class Graph(object):
	"""description of class"""
	def __init__(self, nodes = [], edges = [], symmetric = True):
		self.nodes = nodes
		self.edges = []
		for edge in edges:
			self.add_edge(edge)
		self.build_adjacency_matrix(symmetric)

	def add_node(self, node):
		self.nodes.append(node)

	def add_edge(self, edge):
		if len(edge) != 3: 
			raise ValueError('An edge needs to have three values: starting node, ending node, and the weight (1 for unweighted graph)')
		if edge[0] not in self.nodes:
			raise ValueError('Starting node of the edge is unknown, i.e., not in the node list of the graph')
		if edge[1] not in self.nodes:
			raise ValueError('Ending node of the edge is unknown, i.e., not in the node list of the graph')
		self.edges.append((self.nodes.index(edge[0]), self.nodes.index(edge[1]), edge[2]))

	def build_adjacency_matrix(self, symmetric = True):
		self.adj_mat = np.zeros((len(self.nodes), len(self.nodes)))
		for edge in self.edges:
			self.adj_mat[edge[0]][edge[1]] = edge[2]
			if symmetric:
				self.adj_mat[edge[1]][edge[0]] = edge[2]

	def harmonic_function_label_propagation(self, fixed_indices_vals, rescale_extremes = True, normalize = True):
		self.wedeg_mat = np.zeros((len(self.nodes), len(self.nodes)))
		for i in range(len(self.nodes)):
			self.wedeg_mat[i][i] = sum(self.adj_mat[i])
			
		lap_mat = np.subtract(self.wedeg_mat, self.adj_mat)
		lap_mat_uu = lap_mat[np.ix_([x for x in range(len(self.nodes)) if x not in [y[0] for y in fixed_indices_vals]], [x for x in range(len(self.nodes)) if x not in [y[0] for y in fixed_indices_vals]])]
		lap_mat_ul = lap_mat[np.ix_([x for x in range(len(self.nodes)) if x not in [y[0] for y in fixed_indices_vals]], [y[0] for y in fixed_indices_vals])]
		scores_l = np.expand_dims(np.array([y[1] for y in fixed_indices_vals]), axis = 0)
		
		scores_u = np.dot(np.dot(np.multiply(-1.0, np.linalg.inv(lap_mat_uu)), lap_mat_ul), scores_l.T)
		unlab_docs = [x for x in self.nodes if self.nodes.index(x) not in [y[0] for y in fixed_indices_vals]]
		all_scores = dict(zip(unlab_docs, scores_u.T[0]))
		
		for e in fixed_indices_vals:
			if not rescale_extremes:
				all_scores[self.nodes[e[0]]] = e[1]
			else: 
				adj_row = self.adj_mat[e[0]]
				adj_row = np.multiply(1.0 / np.sum(adj_row), adj_row)
				all_scores[self.nodes[e[0]]] = sum([adj_row[i] * all_scores[self.nodes[i]] for i in range(len(self.nodes)) if i not in [y[0] for y in fixed_indices_vals]]) 

		if normalize:
			min_score = min(all_scores.values())
			max_score = max(all_scores.values())
			for k in all_scores:
				all_scores[k] = (all_scores[k] - min_score) / (max_score - min_score)
		return all_scores
		
	
	def pagerank(self, alpha = 0.15, init_pr_vector = None, fixed_indices = None, rescale_extremes = True):
		#print("Running PageRank...")
		if init_pr_vector is None:
			init_pr_vector = np.expand_dims(np.full((len(self.nodes)), 1.0/((float)(len(self.nodes)))), axis = 0)
		
		# normalization and stochasticity adjustment of the adjacence matrix
		pr_mat = np.zeros((len(self.nodes), len(self.nodes)))
		for i in range(len(self.nodes)):
			if np.count_nonzero(self.adj_mat[i]) == 0:
				pr_mat[i][:] = np.full((len(self.nodes)), 1.0/((float)(len(self.nodes))))
			else:
				pr_mat[i][:] =  np.multiply(1.0 / np.sum(self.adj_mat[i]), self.adj_mat[i])

		# primitivity adjustment
		pr_mat = np.multiply(1 - alpha, pr_mat) + np.multiply(alpha, np.full((len(self.nodes), len(self.nodes)), 1.0/((float)(len(self.nodes)))))

		# pagerank iterations
		diff = 1
		it = 1
		while diff > 0.001:
			old_vec = init_pr_vector
			init_pr_vector = np.dot(init_pr_vector, pr_mat)
			#init_pr_vector = np.multiply(1.0 / np.sum(init_pr_vector), init_pr_vector)
			
			if fixed_indices is not None:
				for ind in fixed_indices:	
					init_pr_vector[0][ind] = fixed_indices[ind]

			diff = np.sum(np.abs(init_pr_vector - old_vec))
			#print("PR iteration " + str(it) + ": " + str(init_pr_vector))
			it += 1
		
		
		if fixed_indices is not None and rescale_extremes:
				for ind in fixed_indices:	
					adj_row = self.adj_mat[ind]
					adj_row = np.multiply(1.0 / np.sum(adj_row), adj_row)
					init_pr_vector[0][ind] = sum([adj_row[i] * init_pr_vector[0][i] for i in range(len(self.nodes)) if i != ind]) 

		return dict(zip(self.nodes, init_pr_vector[0]))
