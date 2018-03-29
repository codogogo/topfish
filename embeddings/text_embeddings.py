import numpy as np
from helpers import io_helper as ioh
import codecs
from helpers import io_helper

def aggregate_phrase_embedding(words, stopwords, embs, emb_size, l2_norm_vec = True, lang = 'en'):
	vec_res = np.zeros(emb_size)
	fit_words = [w.lower() for w in words if w.lower() not in stopwords and w.lower() in embs.lang_vocabularies[lang]]
	if len(fit_words) == 0:
		return None

	for w in fit_words:
		vec_res += embs.get_vector(lang, w) 	
	res = np.multiply(1.0 / (float(len(fit_words))), vec_res)
	if l2_norm_vec:
		res = np.multiply(1.0 / np.linalg.norm(res), res)
	return res


class Embeddings(object):
	"""Captures functionality to load and store textual embeddings"""

	def __init__(self, cache_similarities = False):
		self.lang_embeddings = {}
		self.lang_emb_norms = {}
		self.lang_vocabularies = {}
		self.emb_sizes = {}
		self.cache = {}
		self.do_cache = cache_similarities

	def inverse_vocabularies(self):
		self.inverse_vocabularies = {}
		for l in self.lang_vocabularies:
			self.inverse_vocabularies[l] = {v: k for k, v in self.lang_vocabularies[l].items()}

	def get_word_from_index(self, index, lang = 'en'):
		if index in self.inverse_vocabularies[lang]:
			return self.inverse_vocabularies[lang][index]
		else:
			return None

	def get_vector(self, lang, word):
		if word in self.lang_vocabularies[lang]:
			return self.lang_embeddings[lang][self.lang_vocabularies[lang][word]]
		else: 
			return None

	def set_vector(self, lang, word, vector):
		if word in self.lang_vocabularies[lang]:
			self.lang_embeddings[lang][self.lang_vocabularies[lang][word]] = vector

	def get_norm(self, lang, word):
		if word in self.lang_vocabularies[lang]:
			return self.lang_emb_norms[lang][self.lang_vocabularies[lang][word]]
		else: 
			return None

	def set_norm(self, lang, word, norm):
		if word in self.lang_vocabularies[lang]:
			self.lang_emb_norms[lang][self.lang_vocabularies[lang][word]] = norm

	def add_word(self, lang, word, vector = None):
		if word not in self.lang_vocabularies[lang]:
			self.lang_vocabularies[lang][word] = len(self.lang_vocabularies[lang])
			rvec = np.random.uniform(-1.0, 1.0, size = [self.emb_sizes[lang]]) if vector is None else vector
			rnrm = np.linalg.norm(rvec, 2)
			self.lang_embeddings[lang] = np.vstack((self.lang_embeddings[lang], rvec))
			self.lang_emb_norms[lang] = np.concatenate((self.lang_emb_norms[lang], [rnrm]))	

	def remove_word(self, lang, word):
		self.lang_vocabularies[lang].pop(word, None)
	
	def load_embeddings(self, filepath, limit, language = 'en', print_loading = "False", skip_first_line = False, min_one_letter = False, special_tokens = None):
		vocabulary, embs, norms = ioh.load_embeddings_dict_with_norms(filepath, limit = limit, special_tokens = special_tokens, print_load_progress = print_loading, skip_first_line = skip_first_line, min_one_letter = min_one_letter)		
		self.lang_embeddings[language] = embs
		self.lang_emb_norms[language] = norms
		self.emb_sizes[language] = embs.shape[1]
		self.lang_vocabularies[language] = vocabulary	
	

	def word_similarity(self, first_word, second_word, first_language = 'en', second_language = 'en'):	
		if self.do_cache:
			cache_str = min(first_word, second_word) + "-" + max(first_word, second_word)
			if (first_language + "-" + second_language) in self.cache and cache_str in self.cache[first_language + "-" + second_language]:
				return self.cache[first_language + "-" + second_language][cache_str]
		elif (first_word not in self.lang_vocabularies[first_language] and first_word.lower() not in self.lang_vocabularies[first_language]) or (second_word not in self.lang_vocabularies[second_language] and second_word.lower() not in self.lang_vocabularies[second_language]):
			if ((first_word in second_word or second_word in first_word) and first_language == second_language) or first_word == second_word:
					return 1
			else:
					return 0

		index_first = self.lang_vocabularies[first_language][first_word] if first_word in self.lang_vocabularies[first_language] else (self.lang_vocabularies[first_language][first_word.lower()] if first_word.lower() in self.lang_vocabularies[first_language] else -1)
		index_second = self.lang_vocabularies[second_language][second_word] if second_word in self.lang_vocabularies[second_language] else (self.lang_vocabularies[second_language][second_word.lower()] if second_word.lower() in self.lang_vocabularies[second_language] else -1)		

		if index_first >= 0 and index_second >= 0:		
			first_emb = self.lang_embeddings[first_language][index_first]
			second_emb = self.lang_embeddings[second_language][index_second] 

			first_norm = self.lang_emb_norms[first_language][index_first]
			second_norm = self.lang_emb_norms[second_language][index_second]

			score =  np.dot(first_emb, second_emb) / (first_norm * second_norm)
		else:
			score = 0
		
		if self.do_cache:
			if (first_language + "-" + second_language) not in self.cache:
				self.cache[first_language + "-" + second_language] = {}
				if cache_str not in self.cache[first_language + "-" + second_language]:
					self.cache[first_language + "-" + second_language][cache_str] = score		
		return score

	def most_similar(self, embedding, target_lang, num, similarity = True):
		ms = []
		for w in self.lang_vocabularies[target_lang]:
			targ_w_emb = self.get_vector(target_lang, w)
			if len(embedding) != len(targ_w_emb):
				print("Unaligned embedding length: " + w)
			else:
				if similarity:
					nrm = np.linalg.norm(embedding, 2)
					trg_nrm = self.get_norm(target_lang, w)
					sim = np.dot(embedding, targ_w_emb) / (nrm * trg_nrm)
					if (len(ms) < num) or (sim > ms[-1][1]):	
						ms.append((w, sim))
						ms.sort(key = lambda x: x[1], reverse = True)
				else:
					dist = np.linalg.norm(embedding - targ_w_emb)
					if (len(ms) < num) or (dist < ms[-1][1]):	
						ms.append((w, dist))
						ms.sort(key = lambda x: x[1])
				if len(ms) > num: 
						ms.pop() 
		return [ws for ws in ms]
	
	def merge_embedding_spaces(self, languages, emb_size, merge_name = 'merge', lang_prefix_delimiter = '__', special_tokens = None):
		print("Merging embedding spaces...")
		merge_vocabulary = {}
		merge_embs = []
		merge_norms = []

		for lang in languages:
			print("For language: " + lang)
			norms =[]
			embs = []
			for word in self.lang_vocabularies[lang]:
				if special_tokens is None or word not in special_tokens:
					merge_vocabulary[lang + lang_prefix_delimiter + word] = len(merge_vocabulary)
				else:
					merge_vocabulary[word] = len(merge_vocabulary)
				embs.append(self.get_vector(lang, word))
				norms.append(self.get_norm(lang, word))
			merge_embs =  np.copy(embs) if len(merge_embs) == 0 else np.vstack((merge_embs, embs))
			merge_norms = np.copy(norms) if len(merge_norms) == 0 else np.concatenate((merge_norms, norms))
			
		self.lang_vocabularies[merge_name] = merge_vocabulary
		self.lang_embeddings[merge_name] = merge_embs
		self.lang_emb_norms[merge_name] = merge_norms
		self.emb_sizes[merge_name] = emb_size  

	def store_embeddings(self, path, language):
		io_helper.store_embeddings(path, self, language)