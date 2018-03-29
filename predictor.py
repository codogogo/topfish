from embeddings import text_embeddings
import nlp
from helpers import io_helper
from sts import simple_sts 
from sys import stdin
import argparse
import os
from datetime import datetime
import ast

supported_lang_strings = {"en" : "english", "fr" : "french", "de" : "german", "es" : "spanish", "it" : "italian"}

parser = argparse.ArgumentParser(description='Makes predictions with a pre-trained CNN classifier.')
parser.add_argument('datadir', help='A path to the directory containing the input text files.')
parser.add_argument('embs', help='A path to the file containing pre-trained word embeddings')
parser.add_argument('model', help='A file path to which to store the trained model.')
parser.add_argument('output', help='A file path to which to store the predictions made by the pre-trained model.')
parser.add_argument('-ltf', '--longformat', type = bool, help='Indicates that the supplied text files are in the long format, each text file is one training instance (default = False -- short text format, each line of each file is one training instance)', default = False)
parser.add_argument('-g', '--goldlabels', type = bool, help='Indicates whether the true (gold) class labels are given in the text files (default = True).', default = True)
	
args = parser.parse_args()

if not os.path.isdir(os.path.dirname(args.datadir)):
	print("Error: Directory containing the input files not found.")
	exit(code = 1)

if not os.path.isfile(args.embs):
	print("Error: File containing pre-trained word embeddings not found.")
	exit(code = 1)

if not os.path.isfile(args.model):
	print("Error: File containing pre-trained classification model not found.")
	exit(code = 1)


if not os.path.isdir(os.path.dirname(args.output)) and not os.path.dirname(args.output) == "":
	print("Error: Directory of the provided output file path does not exist.")
	exit(code = 1)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Loading and preprocessing texts.", flush = True)
files = io_helper.load_all_files(args.datadir)
filenames = [x[0] for x in files]
texts = [x[1] for x in files]

wrong_lang = False
if args.longformat:
	languages = [x.split("\t", 1)[0].strip().lower() for x in texts]	
	labels = [x.split("\t", 2)[1].strip().lower() for x in texts] if args.goldlabels else None
	raw_texts = [x.split("\t", 2)[2].strip() for x in texts] if args.goldlabels else [x.split("\t", 1)[1].strip() for x in texts] 
	for i in range(len(languages)):
		if languages[i] not in supported_lang_strings.keys() and languages[i] not in supported_lang_strings.values():
			print("The format of the file is incorrect, unspecified or unsupported language: " + str(filenames[i]))
			wrong_lang = True
else:
	languages = []
	labels = []
	raw_texts = []
	for i in range(len(filenames)):
		langs = [x.split("\t")[0] for x in texts[i].split("\n") if len(x.split("\t")) >= 3]
		for j in range(len(langs)):
			if langs[j] not in supported_lang_strings.keys() and langs[j] not in supported_lang_strings.values():
				print("The format of the file is incorrect, unspecified or unsupported language: " + str(filenames[i] + ", line " + str(j+1)))
				wrong_lang = True
		languages.extend(langs)
		if args.goldlabels:
			labs = [x.split("\t")[1] for x in texts[i].split("\n") if len(x.split("\t")) >= 3]
			labels.extend(labs)
		if args.goldlabels: 
			rtxts = [x.split("\t", 2)[2] for x in texts[i].split("\n") if len(x.split("\t")) >= 3]
		else:
			rtxts = [x.split("\t", 1)[1] for x in texts[i].split("\n") if len(x.split("\t")) >= 2]
		raw_texts.extend(rtxts)
	if not args.goldlabels:
		labels = None
if wrong_lang:
	exit(code = 2)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Loading word embeddings.", flush = True)
embeddings = text_embeddings.Embeddings()
embeddings.load_embeddings(args.embs, limit = 1000000, language = 'default', print_loading = True, skip_first_line = True, special_tokens = ["<PAD/>", "<NUM/>", "<PUNC/>"])

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Loading the pre-trained classification model and making predictions.", flush = True)
parameters = { "batch_size" : 50 }

nlp.test_cnn(raw_texts, languages, labels, embeddings, args.model, args.output, parameters, emb_lang = "default")
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Predictions stored. My job is done here, ciao bella!", flush = True)