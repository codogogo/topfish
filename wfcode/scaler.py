import numpy as np
import math

class LinearScaler(object):
	def __init__(self, items, sims):
		self.items = items
		self.sims = sims

	def scale(self):
		minsim = min([x[2] for x in self.sims])
		minsim_edge = ([x for x in self.sims if x[2] == minsim])[0]		
		ind_first = self.items.index(minsim_edge[0])
		ind_second = self.items.index(minsim_edge[1])

		# linear interpolation scaling
		scales = {}
		for i in range(len(self.items)):
			if i != ind_first and i != ind_second:
				sim_minus = ([x[2] for x in self.sims if (x[0] == minsim_edge[1] and x[1] == self.items[i]) or (x[1] == minsim_edge[1] and x[0] == self.items[i])])[0]
				sim_plus = ([x[2] for x in self.sims if (x[0] == minsim_edge[0] and x[1] == self.items[i]) or (x[1] == minsim_edge[0] and x[0] == self.items[i])])[0]
				scales[self.items[i]] = (-1 * sim_minus) / (sim_minus + sim_plus) + sim_plus / (sim_minus + sim_plus)
			elif i == ind_first:
				scales[self.items[i]] = 1
			elif i == ind_second:
				scales[self.items[i]] = -1
		return scales

class WordfishScaler(object):
	"""implementation of a WordFish-like scaling"""
	def __init__(self, corpus):
		self.corpus = corpus
		self.num_docs = len(self.corpus.docs_raw)
		self.num_words = len(self.corpus.vocabulary)
		
		self.alpha_docs = np.zeros(self.num_docs)
		self.theta_docs = np.zeros(self.num_docs)
		self.beta_words = np.zeros(self.num_words)
		self.psi_words = np.zeros(self.num_words)
		self.log_expectations = np.zeros((self.num_docs, self.num_words))

	def initialize(self):
		print("Initializing...")
		# Setting initial values for word fixed effects (psi)			
		self.psi_words =  np.log(np.average(self.corpus.occurrences, axis = 0))
				
		# Setting initial values for document fixed effects (Alphas) 
		counts = np.sum(self.corpus.occurrences, axis = 1)
		self.alpha_docs = np.log(np.multiply(counts, 1.0 / counts[0]))
		print("Alpha docs: ")
		print(self.alpha_docs)

		# Setting initial values for betas and omegas
		matrix = np.log(np.transpose(self.corpus.occurrences)) - np.transpose(np.repeat(np.expand_dims(self.psi_words, 0), self.num_docs, axis = 0)) - np.repeat(np.expand_dims(self.alpha_docs, 0), self.num_words, axis = 0)
		u, s, v = np.linalg.svd(matrix, full_matrices = False, compute_uv = True)
		self.beta_words = u[:,0]
		self.theta_docs = v[0,:]

	def normalize_positions(self):
		self.alpha_docs[0] = 0
		self.theta_docs = np.divide((self.theta_docs - np.full((1, self.num_docs), np.mean(self.theta_docs))), np.full((1, self.num_docs), np.std(self.theta_docs)))
		self.theta_docs = self.theta_docs[0] 

	def train(self, learning_rate, num_iters):
		print("Training...")
		# Computing the objective and also refreshing lambdas (log-likelihoods) for all pairs of word-document
		self.normalize_positions()
		obj_score = self.objective() 
		print("Initial objective score: " + str(obj_score))

		for i in range(num_iters):
			# Updating document parameters
			alpha_grads, theta_grads = self.gradients_docs()
			self.alpha_docs = self.alpha_docs - np.multiply(alpha_grads, learning_rate / self.num_words)
			self.theta_docs = self.theta_docs - np.multiply(theta_grads, learning_rate / self.num_words)

			self.normalize_positions()

			#obj_score = self.objective() 
			#if i % 100 == 0: print("Iteration (primary) " + str(i+1) + ": " + str(obj_score))
						
			# Updating word parameters
			beta_grads, psi_grads = self.gradients_words()
			self.beta_words = self.beta_words - np.multiply(beta_grads, learning_rate / self.num_docs)
			self.psi_words = self.psi_words - np.multiply(psi_grads, learning_rate / self.num_docs)

			obj_score = self.objective() 
			if i % 100 == 0:
				print("Iteration (secondary) " + str(i+1) + ": " + str(obj_score))

		self.normalize_positions()
		self.corpus.set_doc_positions(self.theta_docs)
		
	def objective(self): 
		self.log_expectations = self.log_expectation()
		return -1 * np.sum(np.multiply(self.corpus.occurrences, self.log_expectations) - np.exp(self.log_expectations))

	def log_expectation(self):
		return np.transpose(np.repeat(np.expand_dims(self.alpha_docs, 0), self.num_words, axis = 0)) + np.repeat(np.expand_dims(self.psi_words, 0), self.num_docs, axis = 0) + np.outer(self.theta_docs, self.beta_words)
		
	def gradients_words(self):
		psi_grads = np.sum(np.exp(self.log_expectations) - self.corpus.occurrences, axis = 0)	
		beta_grads = np.sum(np.multiply(np.exp(self.log_expectations) - self.corpus.occurrences, np.transpose(np.repeat(np.expand_dims(self.theta_docs, 0), self.num_words, axis = 0))), axis = 0)
		return [beta_grads, psi_grads]

	def gradients_docs(self):
		alpha_grads = np.sum(np.exp(self.log_expectations) - self.corpus.occurrences, axis = 1)	
		theta_grads = np.sum(np.multiply(np.exp(self.log_expectations) - self.corpus.occurrences, np.repeat(np.expand_dims(self.beta_words, 0), self.num_docs, axis = 0)), axis = 1)
		return [alpha_grads, theta_grads]