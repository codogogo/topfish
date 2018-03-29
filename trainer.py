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

parser = argparse.ArgumentParser(description='Trains a CNN-based text classification model.')
parser.add_argument('datadir', help='A path to the directory containing the input text files with annotations.')
parser.add_argument('embs', help='A path to the file containing pre-trained word embeddings')
parser.add_argument('model', help='A file path to which to store the trained model.')
parser.add_argument('-ltf', '--longformat', type = bool, help='Indicates that the supplied text files are in the long format, each text file is one training instance (default = False -- short text format, each line of each file is one training instance)', default = False)
parser.add_argument('-l', '--learnrate', type=float, help='Learning rate value (default = 0.0001)', default = 0.0001)
parser.add_argument('-e', '--epochs', type=int, help='Number of training epochs -- in one epoch each training instance is exposed to the classifier once (default 10)', default=10)
parser.add_argument('-r', '--regrate', type=int, help='Parameter regularization factor (L2-norm-based regularization function; default 0.001)', default=0.001)
parser.add_argument('-ue', '--updateembs', type=bool, help='Indicates whether or not to update word embeddings during training (default = False)', default=False)
parser.add_argument('-nl', '--numlayers', type=int, help='Number of layers of the convolutional neural network (one layer = one convolution operator + one max-pooling operator; default 1)', default=1)
parser.add_argument('-f', '--filters', help='Filters and filter sizes (need to be defined for each layer of the CNN. For examples, "[[(3, 16), (4, 32), (5, 16)], [(2, 12), (3, 12)]])" indicates two network layers, first with 16 filters of size 3, 32 filters of size 4, and 16 filters of size 5, and the second layer with 12 filters of size 2 and 12 filters of size 3. (default [[(3, 16), (4, 32), (5, 16)]])')
parser.add_argument('-mp', '--maxpools', help='Max pools per filter, one size for layer. For example, [4, 1] indicates that there are two network layers, in the first layer we pool 4 biggest convolution scores for each filter, and in the second layer 1 largest value for each filter. (default [1])')
parser.add_argument('-b', '--batchsize', type=int, help='The size of the minibatches in which to train the model (default 50)', default=50)
parser.add_argument('-de', '--diffend', type=float, help='The differences in training loss value between two consecutive epochs required to stop the training (if below given value; default 0.001)', default=0.001)
#parser.add_argument('-pt', '--printtrain', type=bool, help='Indicate whether to print the training, including loss values for each training batch (default True)', default=True)
	
args = parser.parse_args()

if not os.path.isdir(os.path.dirname(args.datadir)):
	print("Error: Directory containing the input files not found.")
	exit(code = 1)

if not os.path.isfile(args.embs):
	print("Error: File containing pre-trained word embeddings not found.")
	exit(code = 1)

if not os.path.isdir(os.path.dirname(args.model)) and not os.path.dirname(args.model) == "":
	print("Error: Directory of the desired model file path does not exist.")
	exit(code = 1)

#if args.stopwords and not os.path.isfile(args.stopwords):
#	print("Error: File containing stopwords not found.")
#	exit(code = 1)

filters = ast.literal_eval(args.filters) if args.filters else [[(3, 16), (4, 32), (5, 16)]]
maxpools = ast.literal_eval(args.maxpools) if args.maxpools else [1]

if len(filters) != args.numlayers or len(maxpools) != args.numlayers:
	print("Error: Number of layers must correspond to the number of layers of filters list (number of sublists in the filters list) and the number of pooling layers (number of entries in the -maxpools list).")
	exit(code = 1)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Loading and preprocessing texts.", flush = True)
files = io_helper.load_all_files(args.datadir)
filenames = [x[0] for x in files]
texts = [x[1] for x in files]

wrong_lang = False
if args.longformat:
	languages = [x.split("\t", 1)[0].strip().lower() for x in texts]	
	labels = [x.split("\t", 2)[1].strip().lower() for x in texts]
	raw_texts = [x.split("\t", 2)[2].strip() for x in texts]
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
		labs = [x.split("\t")[1] for x in texts[i].split("\n") if len(x.split("\t")) >= 3]
		labels.extend(labs)
		rtxts = [x.split("\t", 2)[2] for x in texts[i].split("\n") if len(x.split("\t")) >= 3]
		raw_texts.extend(rtxts)

if wrong_lang:
	exit(code = 2)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Loading word embeddings.", flush = True)
embeddings = text_embeddings.Embeddings()
embeddings.load_embeddings(args.embs, limit = 1000000, language = 'default', print_loading = True, skip_first_line = True, special_tokens = ["<PAD/>", "<NUM/>", "<PUNC/>"])

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Starting the training process, with parameters: \n", flush = True)
parameters = { "num_epochs" : args.epochs, "num_convolutions" : args.numlayers, "filters" : filters, "k_max_pools" : maxpools, "reg_factor" : args.regrate, "update_embeddings" : args.updateembs, "learning_rate" : args.learnrate, "batch_size" : args.batchsize, "epoch_diff_smaller_end" : args.diffend }
print(parameters)

nlp.train_cnn(raw_texts, languages, labels, embeddings, parameters, args.model, emb_lang = "default")
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Training finished. Classifier successfully stored.", flush = True)