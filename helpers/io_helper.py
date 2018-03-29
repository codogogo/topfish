from __future__ import division
import codecs
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
from helpers import data_helper
import re

################################################################################################################################

def serialize(item, path):
	pickle.dump(item, open(path, "wb" ))

def deserialize(path):
	return pickle.load(open(path, "rb" ))

def load_file(filepath):
	return (codecs.open(filepath, 'r', encoding = 'utf8', errors = 'replace')).read()

def load_lines(filepath):
	return [l.strip() for l in list(codecs.open(filepath, "r", encoding = 'utf8', errors = 'replace').readlines())]

def load_blocked_lines(filepath):
	lines = [l.strip() for l in list(codecs.open(filepath, "r", encoding = 'utf8', errors = 'replace').readlines())]
	blocks = []
	block = []
	for l in lines:
		if l == "":
			blocks.append(block)
			block = []
		else:
			block.append(l)
	if len(block) > 0:
		blocks.append(block)
	return blocks

def load_all_files(dirpath):
	files = []
	for filename in listdir(dirpath):		
		files.append((filename, load_file(dirpath + "/" + filename)))
	return files

################################################################################################################################

def store_embeddings(path, embeddings, language, print_progress = True):
	f = codecs.open(path,'w',encoding='utf8')
	vocab = embeddings.lang_vocabularies[language]
	embs = 	embeddings.lang_embeddings[language]

	cnt = 0
	for word in vocab:
		cnt += 1
		if print_progress and cnt % 1000 == 0:
			print("Storing embeddings " + str(cnt))
		f.write(word + " ")
		for i in range(len(embs[vocab[word]])):
			f.write(str(embs[vocab[word]][i]) + " ")
		f.write("\n")
	f.close()

def load_embeddings_dict_with_norms(filepath, limit = None, special_tokens = None, print_load_progress = False, min_one_letter = False, skip_first_line = False):
	norms = []
	vocabulary = {}
	embeddings = []
	cnt = 0
	cnt_dict = 0
	emb_size = -1

	with codecs.open(filepath,'r',encoding='utf8', errors='replace') as f:
		for line in f:
			try:
				cnt += 1
				if limit and cnt > limit: 
					break
				if print_load_progress and (cnt % 1000 == 0): 
					print("Loading embeddings: " + str(cnt))
				if cnt > 1 or not skip_first_line:
					splt = line.split()
					word = splt[0]
					if min_one_letter and not any(c.isalpha() for c in word):
						continue

					vec = [np.float32(x) for x in splt[1:]]
					if emb_size < 0 and len(vec) > 10:
						emb_size = len(vec)

					if emb_size > 0 and len(vec) == emb_size:
						vocabulary[word] = cnt_dict
						cnt_dict += 1
						norms.append(np.linalg.norm(vec, 2))
						embeddings.append(vec)			
			except(ValueError,IndexError,UnicodeEncodeError):
				print("Incorrect format line!")
	
	if special_tokens is not None:
		for st in special_tokens:
			vocabulary[st] = cnt_dict
			cnt_dict += 1
			vec = np.array([0.1 * (special_tokens.index(st) + 1)] * emb_size) #np.random.uniform(-1.0, 1.0, size = [emb_size])
			norms.append(np.linalg.norm(vec, 2))
			embeddings.append(vec)

	return vocabulary, np.array(embeddings, dtype = np.float32), norms 

############################################################################################################################

def load_whitespace_separated_data(filepath):
	lines = list(codecs.open(filepath,'r',encoding='utf8', errors='replace').readlines())
	return [[x.strip() for x in l.strip().split()] for l in lines]

def load_tab_separated_data(filepath):
	lines = list(codecs.open(filepath,'r',encoding='utf8', errors='replace').readlines())
	return [[x.strip() for x in l.strip().split('\t')] for l in lines]

def load_wn_concepts_dict(path):
	lines = list(codecs.open(path,'r',encoding='utf8', errors='replace').readlines())
	lcols = {x[0] : ' '.join((x[1].split('_'))[2:-2]) for x in [l.strip().split() for l in lines]}
	return lcols

def load_bless_dataset(path):
	lines = list(codecs.open(path,'r',encoding='utf8', errors='replace').readlines())
	lcols = [(x[0].split('-')[0], x[3].split('-')[0], "1" if x[2] == "hyper" else "0") for x in [l.strip().split() for l in lines]]
	return lcols

def write_list(path, list):
	f = codecs.open(path,'w',encoding='utf8')
	for l in list:
		f.write(l + "\n")
	f.close()

def write_dictionary(path, dictionary, append = False):
	f = codecs.open(path,'a' if append else 'w',encoding='utf8')
	for k in dictionary:
		f.write(str(k) + "\t" + str(dictionary[k]) + "\n")
	f.close()

def load_translation_pairs(filepath):
	lines = list(codecs.open(filepath,'r',encoding='utf8', errors='replace').readlines())
	dataset = []; 
	for line in lines:
		spl = line.split(',')
		srcword = spl[0].strip()
		trgword = spl[1].strip(); 
		if (" " not in srcword.strip()) and  (" " not in trgword.strip()):
			dataset.append((srcword, trgword)); 
	return dataset	

def write_list_tuples_separated(path, list, delimiter = '\t'):
	f = codecs.open(path,'w',encoding='utf8')
	for i in range(len(list)):
		for j in range(len(list[i])):
			if j == len(list[i]) - 1: 
				f.write(str(list[i][j]) + '\n')
			else:
				f.write(str(list[i][j]) + delimiter)  
	f.close()

def store_wordnet_rels(dirpath, relname, pos, lang, instances):
	f = codecs.open(dirpath + "/" + lang + "_" + relname + "_" + pos + ".txt",'w',encoding='utf8')
	for i in instances:
		splt = i.split('::')
		f.write(splt[0].replace("_", " ") + "\t" + splt[1].replace("_", " ") + "\t" + str(instances[i]) + "\n")
	f.close()

def load_csv_lines(path, delimiter = ',', indices = None):
	f = codecs.open(path,'r',encoding='utf8')
	lines = [l.strip().split(delimiter) for l in f.readlines()]
	if indices is None:
		return lines
	else:
		return [sublist(l, indices) for l in lines]

def load_csv_lines_line_by_line(path, delimiter = ',', indices = None, limit = None):
	lines = []
	f = codecs.open(path,'r',encoding='utf8')
	line = f.readline().strip()
	cnt = 1
	while line is not '':
		lines.extend(sublist(line, indices) if indices is not None else line.split(delimiter))
		line = f.readline().strip()
		cnt += 1
		if limit is not None and cnt > limit:
			break
	return lines

def sublist(list, indices):
	sublist = []
	for i in indices:	
		sublist.append(list[i])
	return sublist


############################################################################################################################

def load_sequence_labelling_data(path, delimiter = '\t', indices = None, line_start_skip = None):
	f = codecs.open(path,'r',encoding='utf8')
	lines = [[t.strip() for t in l.split(delimiter)] for l in f.readlines()]
	instances = []
	instance = []
	for i in range(len(lines)):
		if line_start_skip is not None and lines[i][0].startswith(line_start_skip):
			continue
		if len(lines[i]) == 1 and lines[i][0] == "":
			instances.append(instance)
			instance = []
		else:
			if indices is None:
				instance.append(lines[i])
			else:
				instance.append(sublist(lines[i], indices))
	if len(instance) > 0:
		instances.append(instance)
	return instances

def load_classification_data(path, delimiter_text_labels = '\t', delimiter_labels = '\t', line_start_skip = None):
	f = codecs.open(path,'r',encoding='utf8')
	lines = [[t.strip() for t in l.split(delimiter_text_labels)] for l in f.readlines()]
	instances = []
	for i in range(len(lines)):
		if line_start_skip is not None and lines[i][0].startswith(line_start_skip):
			continue
		text = data_helper.clean_str(lines[i][0].strip()).split()
		if delimiter_text_labels == delimiter_labels:
			labels = lines[i][1:]
		else:
			labels = lines[i][1].strip().split(delimiter_labels)
		instances.append((text, labels))
	return instances		

############################################################################################################################
# Applications specific loading
############################################################################################################################

def load_snli_data(path):
	l = load_csv_lines(path, delimiter = '\t', indices = [0, 5, 6])
	l.pop(0)

	labels = [x[0] for x in l]
	premises = [x[1] for x in l]
	implications = [x[2] for x in l]
	
	return premises, implications, labels

