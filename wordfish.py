from helpers import io_helper
from wfcode import corpus
from wfcode import scaler
import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser(description='Trains a model for classifying lexico-semantic relations.')
parser.add_argument('datadir', help='A path to the directory containing the input text files for scaling (one score will be assigned per file).')
parser.add_argument('output', help='A file path to which to store the scaling results.')
parser.add_argument('--stopwords', help='A file to the path containing stopwords')
parser.add_argument('-f', '--freqthold', type=int, help='A frequency threshold -- all words appearing less than -ft times will be ignored (default 2)')
parser.add_argument('-l', '--learnrate', type=float, help='Learning rate value (default = 0.00001)')
parser.add_argument('-t', '--trainiters', type=int, help='Number of optimization iterations (default = 5000)')

args = parser.parse_args()

if args.trainiters:
	niter = args.trainiters
else:
	niter = 5000

if args.learnrate:
	lr = args.learnrate
else:
	lr = 0.00001

if args.freqthold:
	ft = args.freqthold
else:
	ft = 2

if not os.path.isdir(os.path.dirname(args.datadir)):
	print("Error: Directory containing the input files not found.")
	exit(code = 1)

if not os.path.isdir(os.path.dirname(args.output)) and not os.path.dirname(args.output) == "":
	print("Error: Directory of the output file does not exist.")
	exit(code = 1)

if args.stopwords and not os.path.isfile(args.stopwords):
	print("Error: File containing stopwords not found.")
	exit(code = 1)

if args.stopwords:
	stopwords = io_helper.load_file_lines(args.stopwords)
else:
	stopwords = None

files = io_helper.load_all_files(args.datadir)
corp = corpus.Corpus(files)	
corp.tokenize(stopwords = stopwords, freq_treshold = ft)
corp.build_occurrences()

wf_scaler = scaler.WordfishScaler(corp)
wf_scaler.initialize()
wf_scaler.train(learning_rate = lr, num_iters = niter)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " WordFish scaling completed.", flush = True)

scale = []
for x in corp.results:
	scale.append(str(x) + "\t" + str(corp.results[x]))
io_helper.write_list(args.output, scale)
