import numpy as np
import re
import itertools
from collections import Counter
import sys
import codecs
import random
from embeddings import text_embeddings
from sys import stdin

def clean_str(string):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_text_and_labels(path, lowercase = True, multilingual = False, distinct_labels_index = None):
	"""Loads text instances from files (one text one line), splits the data into words and generates labels (as one-hot vectors).
		Returns split sentences and labels.
	"""
    # Load data from files
	lines = [(s.lower() if lowercase else s).strip().split() for s in list(codecs.open(path,'r',encoding='utf8', errors='replace').readlines())]
	x_instances = [l[1:-1] for l in lines] if multilingual else [l[:-1] for l in lines]

	if multilingual: 
		langs = [l[0] for l in lines]
	labels = [l[-1] for l in lines]
	
	dist_labels = list(set(labels)) if distinct_labels_index is None else distinct_labels_index
	y_instances = [np.zeros(len(dist_labels)) for l in labels]
	for i in range(len(y_instances)):
		y_instances[i][dist_labels.index(labels[i])] = 1

	return [x_instances, y_instances, langs, dist_labels] if multilingual else [x_instances, y_instances, dist_labels] 

def build_text_and_labels(texts, class_labels, lowercase = True, multilingual = False, langs = None, distinct_labels_index = None):
	# Load data from files
	lines = [(text.lower() if lowercase else text).strip().split() for text in texts]
	x_instances = [l[1:-1] for l in lines] if multilingual else [l[:-1] for l in lines]
	
	dist_labels = list(set(class_labels)) if distinct_labels_index is None else distinct_labels_index
	y_instances = [np.zeros(len(dist_labels)) for l in class_labels]
	for i in range(len(y_instances)):
		y_instances[i][dist_labels.index(class_labels[i])] = 1

	return [x_instances, y_instances, langs, dist_labels] if multilingual else [x_instances, y_instances, dist_labels] 

def pad_texts(texts, padding_word="<PAD/>", max_length = None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in texts) if max_length is None else max_length
    padded_texts = []
    for i in range(len(texts)):
        text = texts[i]
        num_padding = sequence_length - len(text)
        padded_text = text + [padding_word] * num_padding if num_padding >= 0 else text[ : sequence_length]
        padded_texts.append(padded_text)
    return padded_texts

def build_vocab(texts):
	"""
	Builds a vocabulary mapping from word to index based on the sentences.
	Returns vocabulary mapping and inverse vocabulary mapping.
	"""
	# Build vocabulary
	word_counts = Counter(itertools.chain(*texts))
	# Mapping from index to word
	vocabulary_invariable = [x[0] for x in word_counts.most_common()]
	vocabulary_invariable = list(sorted(vocabulary_invariable))
	# Mapping from word to index
	vocabulary = {x: i for i, x in enumerate(vocabulary_invariable)}
	inverse_vocabulary = {v: k for k, v in vocabulary.items()}
	return [vocabulary, inverse_vocabulary]

def build_input_data(texts, labels, vocabulary, padding_token = "<PAD/>", langs = None, ignore_empty = True):
	x = []
	y = []
	if langs is not None:
		filt_langs = []	
	for i in range(len(texts)):
		num_not_pad = len([x for x in texts[i] if x != padding_token])					
		if (num_not_pad > 0 or (not ignore_empty)):
			x.append([vocabulary[t] for t in texts[i]])
			y.append(labels[i])
			if langs is not None:
				filt_langs.append(langs[i])
	if langs is not None:
		return [np.array(x), np.array(y), langs]
	else:
		return [np.array(x), np.array(y)]

def remove_stopwords(texts, langs, stopwords, lowercase = True, multilingual = False, lang_prefix_delimiter = '__'):
	for i in range(len(texts)):
		texts[i] = [x for x in texts[i] if (x.split('__')[1].strip() if multilingual else x).lower() not in (stopwords[langs[i]] if multilingual else stopwords)]

def filter_against_vocabulary(texts, vocabulary, lowercase = False):
    return [[(t.lower() if lowercase else t) for t in s if (t.lower() if lowercase else t) in vocabulary] for s in texts]

def load_data_build_vocabulary(path, stopwords = None, lowercase = True, multilingual = False, lang_prefix_delimiter = '__'):
	"""
	Loads and preprocesses data.
	Returns input vectors, labels, vocabulary, and inverse vocabulary.
	"""
	# Load and preprocess data
	if multilingual:
		texts, labels, langs, dist_labels = load_text_and_labels(path, lowercase = lowercase, multilingual = True)
		for i in range(len(texts)):
			texts[i] = langs[i].lower() + lang_prefix_delimiter + texts[i]
	else:
		texts, labels, dist_labels = load_text_and_labels(path, lowercase = lowercase, multilingual = False)

	if stopwords is not None:
		texts = remove_stopwords(texts, langs, stopwords, lowercase = lowercase)
	texts_padded = pad_texts(texts)
	
	vocabulary, vocabulary_inverse = build_vocab(texts_padded)
	x, y = build_input_data(texts_padded, labels, vocabulary)
	
	return [x, y, dist_labels, vocabulary, vocabulary_inverse]


def load_data_given_vocabulary(path, vocabulary, stopwords = None, lowercase = False, multilingual = False, lang_prefix_delimiter = '__', max_length = None, split = None, ignore_empty = True, distinct_labels_index = None):
	"""
	Loads and preprocesses data given the vocabulary.
	Returns input vectors, labels, vocabulary, and inverse vocabulary.
	"""
	# Load and preprocess data
	if multilingual:
		texts, labels, langs, dist_labels = load_text_and_labels(path, lowercase = lowercase, multilingual = True, distinct_labels_index = distinct_labels_index)
		for i in range(len(texts)):
			for j in range(len(texts[i])):
				texts[i][j] = langs[i].lower() + lang_prefix_delimiter + texts[i][j]
	else:
		texts, labels, dist_labels = load_text_and_labels(path, lowercase = lowercase, multilingual = False, distinct_labels_index = distinct_labels_index)

	if stopwords is not None:
		remove_stopwords(texts, langs if multilingual else None, stopwords, lowercase = lowercase, multilingual = multilingual)

	texts = filter_against_vocabulary(texts, vocabulary)
	texts_padded = pad_texts(texts, max_length = max_length)
	
	if multilingual:
		x, y, flangs = build_input_data(texts_padded, labels, vocabulary, langs = langs, ignore_empty = ignore_empty)
		dist_langs = set(flangs)
		for dl in dist_langs:
			num = len([l for l in flangs if l == dl])
			print("Language: " + dl + ", num: " + str(num))
	else:
		x, y = build_input_data(texts_padded, labels, vocabulary, ignore_empty = ignore_empty)
	if split is None:
		return [x, y, dist_labels]
	else:
		x_train = x[:split]
		y_train = y[:split]
		x_test = x[split:]
		y_test = y[split:]
		return [x_train, y_train, x_test, y_test, dist_labels]

def build_data_given_vocabulary(data, class_labels, vocabulary, stopwords = None, lowercase = False, multilingual = False, lang_prefix_delimiter = '__', max_length = None, split = None, ignore_empty = True, distinct_labels_index = None):
	"""
	Loads and preprocesses data given the vocabulary.
	Returns input vectors, labels, vocabulary, and inverse vocabulary.
	"""
	# Load and preprocess data
	if multilingual:
		texts, labels, langs, dist_labels = load_text_and_labels(path, lowercase = lowercase, multilingual = True, distinct_labels_index = distinct_labels_index)
		for i in range(len(texts)):
			for j in range(len(texts[i])):
				texts[i][j] = langs[i].lower() + lang_prefix_delimiter + texts[i][j]
	else:
		texts, labels, dist_labels = load_text_and_labels(path, lowercase = lowercase, multilingual = False, distinct_labels_index = distinct_labels_index)

	if stopwords is not None:
		remove_stopwords(texts, langs if multilingual else None, stopwords, lowercase = lowercase, multilingual = multilingual)

	texts = filter_against_vocabulary(texts, vocabulary)
	texts_padded = pad_texts(texts, max_length = max_length)
	
	if multilingual:
		x, y, flangs = build_input_data(texts_padded, labels, vocabulary, langs = langs, ignore_empty = ignore_empty)
		dist_langs = set(flangs)
		for dl in dist_langs:
			num = len([l for l in flangs if l == dl])
			print("Language: " + dl + ", num: " + str(num))
	else:
		x, y = build_input_data(texts_padded, labels, vocabulary, ignore_empty = ignore_empty)
	if split is None:
		return [x, y, dist_labels]
	else:
		x_train = x[:split]
		y_train = y[:split]
		x_test = x[split:]
		y_test = y[split:]
		return [x_train, y_train, x_test, y_test, dist_labels]



def load_vocabulary_embeddings(vocabulary_inv, embeddings, emb_size, padding = "<PAD/>"):
	voc_embs = []
	for i in range(len(vocabulary_inv)):
		if i not in vocabulary_inv:
			raise Exception("Index not in index vocabulary!" + " Index: " + str(i))
		word = vocabulary_inv[i]
		if word == padding:
			voc_embs.append(np.random.uniform(-1.0, 1.0, size = [emb_size]))
		elif word not in embeddings:
			raise Exception("Word not found in embeddings! " + word)
		else:
			 voc_embs.append(embeddings[word])
	return np.array(voc_embs, dtype = np.float32)

def prepare_data_for_kb_embedding(data, prebuilt_dicts = None, valid_triples_dict = None, generate_corrupt = True, num_corrupt = 2):
	if valid_triples_dict is None:
		valid_triples_dict = {}

	if prebuilt_dicts is None:
		cnt_ent = 0
		cnt_rel = 0
		entity_dict = {}
		relations_dict = {}
	else:
		entity_dict = prebuilt_dicts[0]
		relations_dict = prebuilt_dicts[1]

	for d in data:	
		if prebuilt_dicts is None:
			if d[0] not in entity_dict:
				entity_dict[d[0]] = cnt_ent
				cnt_ent += 1
			if d[2] not in entity_dict:
				entity_dict[d[2]] = cnt_ent
				cnt_ent += 1
			if d[1] not in relations_dict:
				relations_dict[d[1]] = cnt_rel
				cnt_rel += 1

		str_rep = str(entity_dict[d[0]]) + "_" + str(relations_dict[d[1]]) + "_" + str(entity_dict[d[2]])
		valid_triples_dict[str_rep] = str_rep

	e1_indices = []
	e2_indices = []
	r_indices = []
	y_vals = []

	count_corrupt_valid = 0
	for d in data:
		e1_ind = entity_dict[d[0]]
		e2_ind = entity_dict[d[2]]
		r_ind = relations_dict[d[1]]

		e1_indices.append(e1_ind)	
		e2_indices.append(e2_ind)
		r_indices.append(r_ind)
		y_vals.append(1)

		if generate_corrupt:
			for i in range(num_corrupt):
				corr_type = random.randint(1,3)
				fake_ind = random.randint(0, (len(entity_dict) if (corr_type == 1 or corr_type == 3) else len(relations_dict)) - 1)
				corr_triple_str_rep =  (str(fake_ind) + "_" + str(r_ind) + "_" + str(e2_ind) if corr_type == 1 else (str(e1_ind) + "_" + str(r_ind) + "_" + str(fake_ind) if corr_type == 3 else str(e1_ind) + "_" + str(fake_ind) + "_" + str(e2_ind)))

				while corr_triple_str_rep in valid_triples_dict:
						fake_ind = random.randint(0, (len(entity_dict) if (corr_type == 1 or corr_type == 3) else len(relations_dict)) - 1)
						corr_triple_str_rep =  (str(fake_ind) + "_" + str(r_ind) + "_" + str(e2_ind) if corr_type == 1 else (str(e1_ind) + "_" + str(r_ind) + "_" + str(fake_ind) if corr_type == 3 else str(e1_ind) + "_" + str(fake_ind) + "_" + str(e2_ind)))
						count_corrupt_valid += 1

				if corr_type == 1:
					e1_indices.append(fake_ind)
					e2_indices.append(e2_ind)
					r_indices.append(r_ind)
				elif corr_type == 2:
					e1_indices.append(e1_ind)
					e2_indices.append(e2_ind)
					r_indices.append(fake_ind)
				elif corr_type == 3:
					e1_indices.append(e1_ind)
					e2_indices.append(fake_ind)
					r_indices.append(r_ind)
				y_vals.append(-1)
	
	return [(entity_dict, relations_dict), valid_triples_dict, np.array(e1_indices, dtype = np.int32), np.array(e2_indices, dtype = np.int32), np.array(r_indices, dtype = np.int32), np.array(y_vals, dtype = np.float32) ]
				
def prepare_wn_data(data, concept_dict, rel_string, rel_string_inv, prev_dict = None):	
	data_out = []	
	if prev_dict is None:
		prev_dict = {}
	
	data = [x for x in data if x[1] == rel_string or x[1] == rel_string_inv]
	
	for i in range(len(data)):	
		d = data[i]
		if d[1] == rel_string:
			rel_str = concept_dict[d[0]] + "_" + concept_dict[d[2]]
			if rel_str not in prev_dict:
				data_out.append((concept_dict[d[0]], concept_dict[d[2]], "1"))			
				prev_dict[rel_str] = 1
		elif d[1] == rel_string_inv: 
			rel_str = concept_dict[d[2]] + "_" + concept_dict[d[0]]
			if rel_str not in prev_dict:
				data_out.append((concept_dict[d[2]], concept_dict[d[0]], "1"))
				prev_dict[rel_str] = 1
	return data_out

def create_corrupts(correct_train, correct_test, concept_dict, prev_dict, num_corrupt = 2, shuffle = True):
	concepts = list(concept_dict.values())
	train_corrupt = []
	test_corrupt = []
	current_dict = {}	

	merged = []
	merged.extend(correct_train)
	merged.extend(correct_test)

	for i in range(len(merged)):
		rel_str = merged[i][1] + "_" + merged[i][0]
		if rel_str not in prev_dict and rel_str not in current_dict:
			(train_corrupt if i < len(correct_train) else test_corrupt).append((merged[i][1], merged[i][0], "0"))
			current_dict[rel_str] = 1
		
		for j in range(num_corrupt - 1):
			c1 = concepts[random.randint(0, len(concepts) - 1)]
			c2 = concepts[random.randint(0, len(concepts) - 1)]
			rel_str = c1 + "_" + c2
			while(rel_str in prev_dict or rel_str in current_dict):
				c1 = concepts[random.randint(0, len(concepts) - 1)]
				c2 = concepts[random.randint(0, len(concepts) - 1)]
				rel_str = c1 + "_" + c2
			(train_corrupt if i < len(correct_train) else test_corrupt).append((c1, c2, "0"))
			current_dict[rel_str] = 1
			
	fdata_train = []
	fdata_train.extend(correct_train)
	fdata_train.extend(train_corrupt)
	
	fdata_test = []
	fdata_test.extend(correct_test)
	fdata_test.extend(test_corrupt)

	if shuffle:
		random.shuffle(fdata_train)
		random.shuffle(fdata_test)
	
	return (fdata_train, fdata_test)
	
def lexically_independent_train_set(data_train, data_test):
	ents_test = [x[0] for x in data_test]
	ents_test.extend([x[1] for x in data_test])
	ents_test = set(ents_test)

	filtered_train = [x for x in data_train if x[0] not in ents_test and x[1] not in ents_test]
	return filtered_train 

def prepare_eval_semrel_emb(word_embeddings, stopwords, emb_size, data, y_direct = False, keep_words = False):
	left_mat = []
	right_mat = []
	gold_labels = []
	words = []

	for i in range(len(data)):
		first_word = data[i][0]
		emb1 = text_embeddings.aggregate_phrase_embedding(first_word.strip().split(), stopwords, word_embeddings, emb_size, l2_norm_vec = False)
		second_word = data[i][1]
		emb2 = text_embeddings.aggregate_phrase_embedding(second_word.strip().split(), stopwords, word_embeddings, emb_size, l2_norm_vec = False)
		
		if emb1 is not None and emb2 is not None:
			left_mat.append(emb1)
			right_mat.append(emb2)
			if keep_words:
				words.append(first_word + '\t' + second_word)
			if not y_direct:
				gold_labels.append(-1.0 if data[i][2] == "0" else 1.0)
			else:
				gold_labels.append(data[i][2])

	if keep_words: 
		return [np.array(left_mat), np.array(right_mat), gold_labels, words]
	else:
		 return [np.array(left_mat), np.array(right_mat), gold_labels]

def prepare_dataset_semrel_emb(entity_dict, selected_embeddings, stopwords, word_embeddings, emb_size, data, dict_examples):
	cnt_ent = len(entity_dict)
	e1_inds = []
	e2_inds = []
	y_vals = []	

	cnt_emb_fail = 0
	cnt_existing = 0
	
	for i in range(len(data)):
		first_word = data[i][0]
		if first_word not in entity_dict:
			emb = text_embeddings.aggregate_phrase_embedding(first_word.strip().split(), stopwords, word_embeddings, emb_size, l2_norm_vec = False)
			if emb is not None:
				selected_embeddings.append(emb)
				entity_dict[first_word] = cnt_ent
				cnt_ent += 1
			else:
				cnt_emb_fail += 1
				continue
		second_word = data[i][1]
		if second_word not in entity_dict:
			emb = text_embeddings.aggregate_phrase_embedding(second_word.strip().split(), stopwords, word_embeddings, emb_size, l2_norm_vec = False)
			if emb is not None:
				selected_embeddings.append(emb)
				entity_dict[second_word] = cnt_ent
				cnt_ent += 1
			else:
				cnt_emb_fail += 1
				continue

		e1i = entity_dict[first_word]
		e2i = entity_dict[second_word]
		stres = str(e1i) + "_" + str(e2i)
		if stres not in dict_examples:
			e1_inds.append(e1i)
			e2_inds.append(e2i)
			y_vals.append(-1.0 if data[i][2] == "0" else 1.0)
			dict_examples[stres] = stres
		else:
			#print("Example (pair of entities) already seen: "+ "\"" + first_word + "\" ; \"" + second_word + "\"")
			cnt_existing += 1

	return [list(zip(e1_inds, e2_inds, y_vals)), selected_embeddings]
	
			

			
				