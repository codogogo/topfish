# -*- coding: utf-8 -*-
import numpy as np
import nltk
import math

def build_tf_idf_indices(texts_tokenized):
	idf_index = {}
	tf_index = {}
	for i in range(len(texts_tokenized)):
		tf_index[i] = {}
		for j in range(len(texts_tokenized[i])):
			w = texts_tokenized[i][j]
			if w not in tf_index[i]:
				tf_index[i][w] = 1
			else:
				tf_index[i][w] += 1
			if w not in idf_index:
				idf_index[w] = {}
			if i not in idf_index[w]:
				idf_index[w][i] = 1
		max_word_freq = max([tf_index[i][x] for x in tf_index[i]])
		print("Max word freq: " + str(max_word_freq))
		for w in tf_index[i]:
			tf_index[i][w] = tf_index[i][w] / max_word_freq
	for w in idf_index:
		idf_index[w] = math.log(len(texts_tokenized) / len(idf_index[w]))
	return tf_index, idf_index

def fix_tokenization(tokens):
	punctuation = [".", ",", "!", ":", "?", ";", "-", ")", "(", "[", "]", "{", "}", "...", "/", "\\", "''", "\"", "'"]
	for i in range(len(tokens)):
		pcs = [p for p in punctuation if tokens[i].endswith(p)]
		if (len(pcs) > 0):
			tokens[i] = tokens[i][:-1]
		pcs = [p for p in punctuation if tokens[i].startswith(p)]
		if (len(pcs) > 0):
			tokens[i] = tokens[i][1:]

def build_vocab(tokens, count_treshold = 1):
	print("Building full vocabulary...")
	full_vocab = {}
	for t in tokens:	
		if t in full_vocab:
			full_vocab[t] = full_vocab[t] + 1
		else:
			full_vocab[t] = 1

	print("Tresholding vocabulary...")
	vocab = [x for x in full_vocab if full_vocab[x] >= count_treshold]
	print("Vocabulary length: " + str(len(vocab)))
	print("Building index dicts...")
	dict = { x : vocab.index(x) for x in vocab }
	inv_dict = { vocab.index(x) : x for x in vocab }
	print("Building count dict...")	
	counts = {  vocab.index(x) : full_vocab[x] for x in vocab }
	
	return (dict, inv_dict, counts)

def simple_tokenize(text, stopwords, lower = False, lang_prefix = None):
	print("Tokenizing text...")
	punctuation = [".", ",", "!", ":", "?", ";", "-", ")", "(", "[", "]", "{", "}", "...", "/", "\\", "''", "\"", "'"]
	toks = [(x.strip().lower() if lower else x.strip()) for x in nltk.word_tokenize(text) if x.strip().lower() not in stopwords and x.strip().lower() not in punctuation]
	fix_tokenization(toks)
	
	if lang_prefix:
		toks = [lang_prefix + "__" + x for x in toks]
	return toks

def aggregate_weighted_text_embedding(embeddings, tf_index, idf_index, lang = "default", weigh_idf = True):
	agg_vec = np.zeros(embeddings.emb_sizes[lang])
	for t in tf_index:
		emb = embeddings.get_vector(lang, t)
		if emb is not None:
			if weigh_idf:
				weight = tf_index[t] * idf_index[t]
			else:
				weight = tf_index[t]
			agg_vec = np.add(agg_vec, np.multiply(weight, emb))
	return agg_vec

def word_movers_distance(embeddings, first_tokens, second_tokens):
	return embeddings.wmdistance(first_tokens, second_tokens)

def greedy_alignment_similarity(embeddings, first_doc, second_doc, lowest_sim = 0.3, length_factor = 0.1):		
		print("Greedy aligning...")
		first_vocab, first_vocab_inv, first_counts_cpy = first_doc
		second_vocab, second_vocab_inv, second_counts_cpy = second_doc
	
		if len(first_vocab) == 0 or len(second_vocab) == 0:
			return 0

		first_counts = {x : first_counts_cpy[x] for x in first_counts_cpy }
		second_counts = {x : second_counts_cpy[x] for x in second_counts_cpy}

		#print("Computing actual document lengths...")		
		len_first = sum(first_counts_cpy.values())
		len_second = sum(second_counts_cpy.values())
		
		# similarity matrix computation
		matrix = np.zeros((len(first_vocab), len(second_vocab)))
		print("Computing the similarity matrix...")
		#print("Vocab. size first: " + str(len(first_vocab)) + "Vocab. size second: " + str(len(second_vocab)))
		cntr = 0
		for ft in first_vocab:
			cntr += 1
			#if cntr % 10 == 0:
			#	print("First vocab, item: " + str(cntr))
			first_index = first_vocab[ft]
			for st in second_vocab:
				second_index = second_vocab[st]
				sim = embeddings.word_similarity(ft, st, "default", "default")
				#print("Embedding similarity, " + ft + "::" + st + ": " + str(sim))
				matrix[first_index, second_index] = sim

		# greedy alignment
		print("Computing the alignment...")
		greedy_align_sum = 0.0		
		counter_left_first = len_first
		counter_left_second = len_second
		tok_to_align = min(counter_left_first, counter_left_second)
		while counter_left_first > 0 and counter_left_second > 0:
			new_tok_to_align = min(counter_left_first, counter_left_second)
			if new_tok_to_align == tok_to_align or (tok_to_align - new_tok_to_align > 10000):
				tok_to_align = new_tok_to_align
				print("Left tokens to align: " + str(tok_to_align))
			ind = np.argmax(matrix.flatten())
			ind_src = ind // matrix.shape[1]
			ind_trg =  ind % matrix.shape[1]

			simil = matrix[ind_src, ind_trg]
			#print("Similarity: " + str(simil))

			#print("Index src: " + str(ind_src) + ", word src: " + str(first_vocab_inv[ind_src].encode(encoding='UTF-8', errors='ignore')))
			#print("Index trg: " + str(ind_trg) + ", word trg: " + str(second_vocab_inv[ind_trg].encode(encoding='UTF-8', errors='ignore')))

			if simil < lowest_sim:
				break;

			min_freq = min(first_counts[ind_src], second_counts[ind_trg])
			greedy_align_sum += simil * min_freq
			matrix[ind_src, ind_trg] = -2
			
			first_counts[ind_src] = first_counts[ind_src] - min_freq	
			second_counts[ind_trg] = second_counts[ind_trg] - min_freq
			
			if first_counts[ind_src] == 0:
				matrix[ind_src, :] = -2
			if second_counts[ind_trg] == 0:
				matrix[:, ind_trg] = -2

			counter_left_first = counter_left_first - min_freq
			counter_left_second = counter_left_second - min_freq
				
		prec = greedy_align_sum / min(len_first, len_second)
		rec = greedy_align_sum / max(len_first, len_second)
		return (((1 - length_factor) * prec) + (length_factor * rec))