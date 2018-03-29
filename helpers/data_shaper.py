from helpers import io_helper
import numpy as np
import re

def punctuation():
	return ['—', '-', '.', ',', ';', ':', '\'', '"', '{', '}', '(', ')', '[', ']']

def is_number(token):
	return re.match('^[\d]+[,]*.?\d*$', token) is not None

def decode_predictions(labels, predictions, flatten = False):
	if len(predictions.shape) == 2:
		labs = [labels[np.nonzero(instance)[0][0]] if len(np.nonzero(instance)[0]) > 0 else '' for instance in predictions]
	elif len(predictions.shape) == 3:
		labs = [[labels[np.nonzero(instance)[0][0]] if len(np.nonzero(instance)[0]) > 0 else '' for instance in sequence] for sequence in predictions]
		if flatten:
			labs = [item for sublist in labs for item in sublist]
	else:
		raise ValueError("Not supported. Only list of single instances or list of sequences supported for decoding labels.")
	return labs

def prep_labels_one_hot_encoding(labels, dist_labels = None, multilabel = False):
	if dist_labels is None:
		if multilabel:
			dist_labels = list(set([y for s in labels for y in s]))
		else:
			dist_labels = list(set(labels))
	y = []
	for i in range(len(labels)):
		lab_vec = [0] * len(dist_labels)
		if multilabel:
			for j in range(len(labels[i])):
				lab_vec[dist_labels.index(labels[i][j])] = 1.0
		else:
			lab_vec[dist_labels.index(labels[i])] = 1.0
		y.append(lab_vec)
	return np.array(y, dtype = np.float64), dist_labels

def prep_word_tuples(word_lists, embeddings, embeddings_language, langs = None, labels = None,):
	examples = []
	if labels: 
		labs = []	
	for i in range(len(word_lists)):
		example = []
		add_example = True
		for j in range(len(word_lists[i])):
			w = ("" if langs is None else langs[i] + "__") + word_lists[i][j]
			if w in embeddings.lang_vocabularies[embeddings_language]:
				example.append(embeddings.lang_vocabularies[embeddings_language][w])	
			elif w.lower() in embeddings.lang_vocabularies[embeddings_language]:
				example.append(embeddings.lang_vocabularies[embeddings_language][w.lower()])
			else:
				add_example = False
				break
		if add_example:
			examples.append(example)
			if labels:	
				labs.append(labels[i])	
	if labels: 
		return examples, labs
	else:
		return examples
			
def prep_sequence_labelling(texts, labels, embeddings, stopwords = None, embeddings_language = 'en', multilingual_langs = None, lowercase = False, pad = True, pad_token = '<PAD/>', numbers_token = None, punct_token = None, dist_labels = None, max_seq_len = None, add_missing_tokens = False):
	x = []
	if labels:
		y = []
	
	for i in range(len(texts)):
		if i % 100 == 0:
			print("Line: " + str(i) + " of " + str(len(texts)))
		tok_list = []
		if labels:
			lab_list = []
		language = embeddings_language if multilingual_langs is None else multilingual_langs[i]

		for j in range(len(texts[i])):
			token_clean = texts[i][j].lower() if lowercase else texts[i][j]
			token = token_clean if multilingual_langs is None else multilingual_langs[i] + "__" + token_clean

			if token_clean.strip() in punctuation() and punct_token is not None:
				token = punct_token
			if is_number(token_clean) and numbers_token is not None:
				token = numbers_token
				
			if stopwords is not None and (token_clean in stopwords[language] or token_clean.lower() in stopwords[language]):
				continue
			if token not in embeddings.lang_vocabularies[embeddings_language] and token.lower() not in embeddings.lang_vocabularies[embeddings_language]:
				if add_missing_tokens: 
					embeddings.add_word(embeddings_language, token)
				else:
					continue

			tok_list.append(embeddings.lang_vocabularies[embeddings_language][token] if token in embeddings.lang_vocabularies[embeddings_language] else embeddings.lang_vocabularies[embeddings_language][token.lower()])
			if labels:
				lab_list.append(labels[i][j])
		x.append(tok_list)
		if labels:
			y.append(lab_list)

	if labels:
		y_clean = []
		if dist_labels is None:
			dist_labels = list(set([l for txt_labs in y for l in txt_labs]))
		for i in range(len(y)):
			lab_list = []
			for j in range(len(y[i])):
				lab_vec = [0] * len(dist_labels)
				lab_vec[dist_labels.index(y[i][j])] = 1.0
				lab_list.append(lab_vec)
			y_clean.append(lab_list)		

	if pad:
		ind_pad = embeddings.lang_vocabularies[embeddings_language][pad_token]
		max_len = max([len(t) for t in x]) if max_seq_len is None else max_seq_len
		x = [t + [ind_pad] * (max_len - len(t)) for t in x]
		if labels:
			for r in y_clean:
				extension = [[0] * len(dist_labels)] * (max_len - len(r))
				r.extend(extension)
		sent_lengths = [len([ind for ind in txt if ind != ind_pad]) for txt in x]
	else:
		sent_lengths = [len(txt) for txt in x]

	if labels:
		return np.array(x, dtype = np.int32), np.array(y_clean, dtype = np.float64), dist_labels, sent_lengths
	else:
		return np.array(x, dtype = np.int32), sent_lengths
	
def prep_classification(texts, labels, embeddings, stopwords = None, embeddings_language = 'en', multilingual_langs = None, lowercase = False, pad = True, pad_token = '<PAD/>', numbers_token = None, punct_token = None, dist_labels = None, max_seq_len = None, add_out_of_vocabulary_terms = False):
	x = []
	y = []
	
	for i in range(len(texts)):
		tok_list = []
		lab_list = []
		language = embeddings_language if multilingual_langs is None else multilingual_langs[i]

		for j in range(len(texts[i])):
			token_clean = texts[i][j].lower() if lowercase else texts[i][j]
			token = token_clean if multilingual_langs is None else multilingual_langs[i] + "__" + token_clean

			if token_clean.strip() in punctuation() and punct_token is not None:
				token = punct_token
			if is_number(token_clean) and numbers_token is not None:
				token = numbers_token
				
			if stopwords is not None and (token_clean in stopwords[language] or token_clean.lower() in stopwords[language]):
				continue
			if token not in embeddings.lang_vocabularies[embeddings_language] and token.lower() not in embeddings.lang_vocabularies[embeddings_language]:
				if add_out_of_vocabulary_terms:
					embeddings.add_word(embeddings_language, token)
				else:
					continue
			if max_seq_len is None or len(tok_list) < max_seq_len:
				tok_list.append(embeddings.lang_vocabularies[embeddings_language][token] if token in embeddings.lang_vocabularies[embeddings_language] else embeddings.lang_vocabularies[embeddings_language][token.lower()])
			else: 
				break
		x.append(tok_list)

	if labels is not None:
		if dist_labels is None:
			dist_labels = list(set([l for txt_labs in labels for l in txt_labs]))
		for i in range(len(labels)):
			lab_vec = [0] * len(dist_labels)
			for j in range(len(labels[i])):
				lab_vec[dist_labels.index(labels[i][j])] = 1.0
			y.append(lab_vec)

	if pad:
		ind_pad =  embeddings.lang_vocabularies[embeddings_language][pad_token]
		max_len = max([len(t) for t in x]) if max_seq_len is None else max_seq_len
		x = [t + [ind_pad] * (max_len - len(t)) for t in x]

	if labels is not None: 
			x_ret = np.array(x, dtype = np.int32)
			y_ret = np.array(y, dtype = np.float64)
			return x_ret, y_ret, dist_labels
	else:
		return np.array(x, dtype = np.int32)

def prepare_contrastive_learning_examples(positives, negatives, num_negatives_per_positive):
	if len(negatives) != len(positives) * num_negatives_per_positive:
		raise ValueError("The number of negative examples (per positive examples) is incorrect!")
	examples = []
	for i in len(positives):
		examples.append(positives[i])
		examples.extend(negatives[i*num_negatives_per_positive : (i+1)*num_negatives_per_positive])
	return examples
