from embeddings import text_embeddings
import nlp
from helpers import io_helper
from sts import simple_sts 
from sys import stdin
import argparse
import os
from datetime import datetime

supported_lang_strings = {"en" : "english", "fr" : "french", "de" : "german", "es" : "spanish", "it" : "italian"}

parser = argparse.ArgumentParser(description='Performs topical text scaling -- uses the pre-trained classifier to topically segment the texts and then performs text scaling independently for each of the topics.')
parser.add_argument('datadir', help='A path to the directory containing the input text files for scaling (one score will be assigned per file).')
parser.add_argument('embs', help='A path to the file containing pre-trained word embeddings')
parser.add_argument('model', help='A file path to which to store the trained model.')
parser.add_argument('output', help='Output directory to which to store the topical scaling results and topically segmented texts.')
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

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Loading word embeddings.", flush = True)
embeddings = text_embeddings.Embeddings()
embeddings.load_embeddings(args.embs, limit = 1000000, language = 'default', print_loading = True, skip_first_line = True, special_tokens = ["<PAD/>", "<NUM/>", "<PUNC/>"])

parameters = { "batch_size" : 50 }

nlp.topically_scale(filenames, texts, langs, embeddings, args.model, args.output, parameters, emb_lang = "default", stopwords = stopwords)
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Topical scaling completed.", flush = True)