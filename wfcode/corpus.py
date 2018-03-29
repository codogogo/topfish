import nltk
import numpy as np
import math
from scipy import spatial
import time
from sys import stdin
from datetime import datetime

class Corpus(object):
	"""description of class"""
	def __init__(self, documents, docpairs = None):
		print("Loading corpus, received: " + str(len(documents)) + " docs.")
		self.docs_raw = [d[1] for d in documents]
		self.docs_names = [d[0] for d in documents]
		self.punctuation = [".", ",", "!", ":", "?", ";", "-", ")", "(", "[", "]", "{", "}", "...", "/", "\\", u"``", "''", "\"", "'", "-", "$" ]
		self.doc_pairs = docpairs
		self.results = {}

	def tokenize(self, stopwords = None, freq_treshold = 5):
		self.stopwords = stopwords
		print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Preprocessing corpus...", flush = True)
		self.docs_tokens = [[tok.strip() for tok in nltk.word_tokenize(doc) if tok.strip() not in self.punctuation and len(tok.strip()) > 2] for doc in self.docs_raw]
		self.freq_dicts = []
		if self.stopwords is not None:
			for i in range(len(self.docs_tokens)):
				self.docs_tokens[i] = [tok.strip() for tok in self.docs_tokens[i] if tok.strip().lower() not in self.stopwords]	
					
	def build_occurrences(self):
		print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Building vocabulary...", flush = True)
		self.vocabulary = {} 
		for dt in self.docs_tokens:
			for t in dt:
				if t not in self.vocabulary:
					self.vocabulary[t] = len(self.vocabulary)

		print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Building coocurrence matrix...", flush = True)
		self.occurrences = np.ones((len(self.docs_tokens), len(self.vocabulary)), dtype = np.float32)
		cnt = 0
		for i in range(len(self.docs_tokens)):
			cnt += 1
			print(str(cnt) + "/" + str(len(self.docs_tokens)))
			for j in range(len(self.docs_tokens[i])):
				word = self.docs_tokens[i][j]
				self.occurrences[i][self.vocabulary[word]] += 1
		if np.isnan(self.occurrences).any():
			raise ValueError("NaN in self.occurrences")

	def set_doc_positions(self, positions):
		for i in range(len(self.docs_names)):
			self.results[self.docs_names[i]] = positions[i]

	#def compute_semantic_similarities_aggregation(self, aggreg_sim_func, embeddings):
	#	sims = []
	#	if self.doc_pairs is None:
	#		for i in range(len(self.docs_names) - 1):
	#			for j in range(i + 1, len(self.docs_names)):
	#				score = aggreg_sim_func(self.freq_dicts[i], self.freq_dicts[j], embeddings, self.docs_langs[i], self.docs_langs[j])
	#				sims.append((self.docs_names[i], self.docs_names[j], score)) 
	#	else:
	#		for dp in self.doc_pairs:
	#			i = self.docs_names.index(dp[0])
	#			j = self.docs_names.index(dp[1])
	#			score = aggreg_sim_func(self.freq_dicts[i], self.freq_dicts[j], embeddings, self.docs_langs[i], self.docs_langs[j])
	#			sims.append((self.docs_names[i], self.docs_names[j], score)) 

	#	self.raw_sims = sims
	#	print("\n Sorted semantic similarities, s1: ")
	#	self.raw_sims.sort(key=lambda tup: tup[2])
	#	for s in self.raw_sims:
	#		print(s[0], s[1], str(s[2]))
			
#	def compute_semantic_similarities(self, doc_similarity_function, sent_similarity_function, embedding_similarity_function):
#		sims = []
#		#tasks = []		
#		start_time = time.time()
#		if self.doc_pairs is None:
#			for i in range(len(self.docs_names) - 1):
#				for j in range(i + 1, len(self.docs_names)):
#					#tasks.append((i, j))
#					#print(self.docs_names[i], self.docs_names[j])
#					score = doc_similarity_function(self.freq_dicts[i], self.freq_dicts[j], sent_similarity_function, embedding_similarity_function, self.docs_langs[i], self.docs_langs[j])
#					#sst = SemSimThread(doc_similarity_function, sent_similarity_function, embedding_similarity_function, self.docs_tokens[i], self.docs_tokens[j], self.docs_langs[i], self.docs_langs[j], "Thread-" + self.docs_names[i] + "-" + self.docs_names[j], 1)
#					#sst.start()
#					#sst.join()
#					sims.append((self.docs_names[i], self.docs_names[j], score[0], score[1]))
#					#print("Similarity: " + str(sst.result))
#		else:
#			for dp in self.doc_pairs:
#				i = self.docs_names.index(dp[0])
#				j = self.docs_names.index(dp[1])
#				print("Measuring similarity: " + dp[0] + " :: " + dp[1])
#				score = doc_similarity_function(self.freq_dicts[i], self.freq_dicts[j], sent_similarity_function, embedding_similarity_function, self.docs_langs[i], self.docs_langs[j])
#				print("Score: " + str(score[0]) + "; " + str(score[1]) + "; " + str(score[2]))
#				sims.append((self.docs_names[i], self.docs_names[j], score))

#		end_time = time.time()
#		print("Time elapsed: " + str(end_time-start_time))
		
#		#num_parallel = 10
#		#num_batches = math.ceil((1.0*len(tasks)) / (1.0*num_parallel))
#		#for i in range(num_batches):
#		#	start_time = time.time()
#		#	print("Batch: " + str(i+1) + "/" + str(num_batches))
#		#	start_range = i * num_parallel
#		#	end_range = (i+1)*num_parallel if (i+1)*num_parallel < len(tasks) else len(tasks)
#		#	threads = [SemSimThread(doc_similarity_function, sent_similarity_function, embedding_similarity_function, self.docs_tokens[task[0]], self.docs_tokens[task[1]], self.docs_langs[task[0]], self.docs_langs[task[1]], self.docs_names[task[0]], self.docs_names[task[1]]) for task in tasks[start_range:end_range]]
#		#	for thr in threads:
#		#		thr.start()
#		#	for thr in threads:
#		#		thr.join()
#		#	print("Thread results: ")
#		#	for thr in threads:
#		#		print(thr.threadID + " " + str(thr.result))	
#		#		sims.append((thr.first_name, thr.second_name, thr.result))
#		#	end_time = time.time()
#		#	print("Time elapsed: " + str(end_time-start_time))
		

#		#sim = parallel(delayed(doc_similarity_function)) doc_similarity_function(self.docs_tokens[i], self.docs_tokens[j], sent_similarity_function, embedding_similarity_function, self.docs_langs[i], self.docs_langs[j])
#		#sims = [(self.docs_names[tasks[i][0]], self.docs_names[tasks[i][1]], threads[i].result) for i in range(len(tasks))]

#		#min_sim = min([x[2] for x in sims])
#		#max_sim = max([x[2] for x in sims])
#		self.raw_sims = sims
#		#self.similarities = [(x[0], x[1], (x[2] - min_sim)/(max_sim - min_sim)) for x in sims]

#		print("\n Sorted semantic similarities, s1: ")
#		#self.raw_sims.sort(key=lambda tup: tup[2][0])
#		for s in self.raw_sims:
#			print(s[0], s[1], str(s[2][0]), str(s[2][1]), str(s[2][2]))

#	def compute_term_similarities(self):
#		sims = []
#		for i in range(len(self.docs_names) - 1):
#			for j in range(i + 1, len(self.docs_names)):
#				print(self.docs_names[i], self.docs_names[j])
#				sim = 1 - spatial.distance.cosine(self.tf_idf_vectors[i], self.tf_idf_vectors[j])
#				print("Term-based similarity: " + str(sim))
#				sims.append((self.docs_names[i], self.docs_names[j], sim))
#		min_sim = min([x[2] for x in sims])
#		max_sim = max([x[2] for x in sims])
#		self.raw_sims = sims
#		self.similarities = [(x[0], x[1], (x[2] - min_sim)/(max_sim - min_sim)) for x in sims]

#		print("\n Sorted tf-idf similarities: ")
#		self.raw_sims.sort(key=lambda tup: tup[2])
#		for s in self.raw_sims:
#			print(s[0], s[1], str(s[2]))

#def most_dissimilar_vector(nodes, edges):
#	vec = []
#	min_score = min([x[2] for x in edges])
#	min_pair = ([x for x in edges if x[2] == min_score])[0]
#	first_added = False
#	for i in range(len(nodes)):
#		if nodes[i] == min_pair[0] or nodes[i] == min_pair[1]:
#			vec.append(-1 if first_added else 1)
#			if not first_added:
#				first_added = True 
#		else: 
#			vec.append(0)
#	return vec
				
				
					 
		
			


