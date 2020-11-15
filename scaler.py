from embeddings import text_embeddings
import nlp
from helpers import io_helper
from sts import simple_sts 
from sys import stdin
import argparse
import os
from datetime import datetime

supported_lang_strings = {"en" : "english", "fr" : "french", "de" : "german", "es" : "spanish", "it" : "italian"}

parser = argparse.ArgumentParser(description='Performs text scaling (assigns a score to each text on a linear scale).')
parser.add_argument('datadir', help='A path to the directory containing the input text files for scaling (one score will be assigned per file).')
parser.add_argument('embs', help='A path to the file containing pre-trained word embeddings')
parser.add_argument('output', help='A file path to which to store the scaling results.')
parser.add_argument('emb_cutoff', help='Length of the embedding-dictionary to use.')
parser.add_argument('--stopwords', help='A file to the path containing stopwords')

args = parser.parse_args()

if not os.path.isdir(os.path.dirname(args.datadir)):
	print("Error: Directory containing the input files not found.")
	exit(code = 1)

if not os.path.isfile(args.embs):
	print("Error: File containing pre-trained word embeddings not found.")
	exit(code = 1)

if not os.path.isdir(os.path.dirname(args.output)) and not os.path.dirname(args.output) == "":
	print("Error: Directory of the output file does not exist.")
	exit(code = 1)

if not os.path.isdir(os.path.dirname(args.emb_cutoff)) and not os.path.dirname(args.emb_cutoff) == "":
	print("Error: Number of embeddings-cutoff does not exist.")
	exit(code = 1)
    
if args.stopwords and not os.path.isfile(args.stopwords):
	print("Error: File containing stopwords not found.")
	exit(code = 1)

files = io_helper.load_all_files(args.datadir)
if len(files) < 4:
	print("Error: There need to be at least 4 texts for a meaningful scaling.")
	exit(code = 1)

filenames = [x[0] for x in files]
texts = [x[1] for x in files]

wrong_lang = False
languages = [x.split("\n", 1)[0].strip().lower() for x in texts]
texts = [x.split("\n", 1)[1].strip().lower() for x in texts]
for i in range(len(languages)):
	if languages[i] not in supported_lang_strings.keys() and languages[i] not in supported_lang_strings.values():
		print("The format of the file is incorrect, unspecified or unsupported language: " + str(filenames[i]))
		wrong_lang = True
if wrong_lang:
	exit(code = 2)

langs = [(l if l in supported_lang_strings.values() else supported_lang_strings[l]) for l in languages]

if args.stopwords:
	stopwords = io_helper.load_lines(args.stopwords)
else:
	stopwords = []

predictions_serialization_path = args.output

embeddings = text_embeddings.Embeddings()
embeddings.load_embeddings(args.embs, limit = int(args.emb_cutoff), language = 'default', print_loading = True, skip_first_line = True)
nlp.scale_efficient(filenames, texts, langs, embeddings, predictions_serialization_path, stopwords)
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Scaling completed.", flush = True)