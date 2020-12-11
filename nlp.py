import tensorflow as tf
import numpy as np
from embeddings import text_embeddings
from helpers import io_helper
from helpers import data_shaper
from convolution import cnn
from ml import loss_functions
from evaluation import confusion_matrix
from ml import trainer
from helpers import data_helper
import math
import nltk
from sts import simple_sts
from graphs import graph
import math
import os
from sys import stdin
from datetime import datetime
from scipy import spatial
import codecs
import pickle

def map_lang(lang):
	if lang.lower() == 'english':
		return 'en'
	elif lang.lower() == 'french':
		return 'fr'
	elif lang.lower() == 'german':
		return 'de'
	elif lang.lower() == 'italian':
		return 'it'
	elif lang.lower() == 'spanish':
		return 'es'
	elif lang.lower() in ["en", "es", "de", "fr", "it"]:
		return lang.lower()
	else:
		return None

def inverse_map_lang(lang):
	if lang.lower() == 'en':
		return 'english'
	elif lang.lower() == 'fr':
		return 'french'
	elif lang.lower() == 'de':
		return 'german'
	elif lang.lower() == 'it':
		return 'italian'
	elif lang.lower() == 'es':
		return 'spanish'
	elif lang.lower() in ["english", "spanish", "german", "french", "italian"]:
		return lang.lower()
	else:
		return None

def load_embeddings(path):
	embeddings = text_embeddings.Embeddings()
	embeddings.load_embeddings(path, limit = None, language = 'default', print_loading = False)
	return embeddings

def build_feed_dict_func(model, data, config = None, predict = False):
	x, y = zip(*data)
	fd = model.get_feed_dict(x, None if None in y else y, 1.0 if predict else 0.5)		
	return fd, y

def eval_func(golds, preds, params = None):
	gold_labs = np.argmax(golds, axis = 1)
	pred_labs = np.argmax(preds, axis = 1)   

	conf_matrix = confusion_matrix.ConfusionMatrix(params["dist_labels"], pred_labs, gold_labs, False, class_indices = True)
	res = conf_matrix.accuracy
	return 0 if math.isnan(res) else res

def get_prediction_labels(preds, dist_labels):
	pred_labs = [dist_labels[x] for x in np.argmax(preds, axis = 1)]   
	return pred_labs
	
def train_cnn(texts, languages, labels, embeddings, parameters, model_serialization_path, emb_lang = 'default'):
	# preparing texts
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Preparing texts...', flush=True)
	texts_clean = [data_helper.clean_str(t.strip()).split() for t in texts]
	# encoding languages (full name to abbreviation)
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Encoding languages (full name to abbreviation)...', flush=True)
	langs = [map_lang(x) for x in languages]
	# preparing training examples
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Preparing training examples...', flush=True)
	x_train, y_train, dist_labels = data_shaper.prep_classification(texts_clean, labels, embeddings, embeddings_language = emb_lang, multilingual_langs = langs, numbers_token = '<NUM/>', punct_token = '<PUNC/>', add_out_of_vocabulary_terms = False)	

	# defining the CNN model
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Defining the CNN model...', flush=True)
	cnn_classifier = cnn.CNN(embeddings = (embeddings.emb_sizes[emb_lang], embeddings.lang_embeddings[emb_lang]), num_conv_layers = parameters["num_convolutions"], filters = parameters["filters"], k_max_pools = parameters["k_max_pools"], manual_features_size = 0)
	cnn_classifier.define_model(len(x_train[0]), len(dist_labels), loss_functions.softmax_cross_entropy, len(embeddings.lang_vocabularies[emb_lang]), l2_reg_factor = parameters["reg_factor"], update_embeddings = parameters["update_embeddings"])
	cnn_classifier.define_optimization(learning_rate = parameters["learning_rate"])
	cnn_classifier.set_distinct_labels(dist_labels)

	# initializing a Tensorflow session
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Initializing a Tensorflow session...', flush=True)
	session = tf.InteractiveSession()
	session.run(tf.global_variables_initializer())

	# training the model
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Training the model...', flush=True)
	simp_trainer = trainer.SimpleTrainer(cnn_classifier, session, build_feed_dict_func, eval_func, configuration_func = None)
	simp_trainer.train(list(zip(x_train, y_train)), parameters["batch_size"], parameters["num_epochs"], num_epochs_not_better_end = 5, epoch_diff_smaller_end = parameters["epoch_diff_smaller_end"], print_batch_losses = True, eval_params = { "dist_labels" : dist_labels })

	# storing the model
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Storing the model...', flush=True)
	cnn_classifier.serialize(session, model_serialization_path)
	session.close()
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Training model is done!', flush=True)

def test_cnn(texts, languages, labels, embeddings, model_serialization_path, predictions_file_path, parameters, emb_lang = 'default'):
	# loading the serialized 
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Loading the serialized model...', flush=True)
	cnn_classifier, session = cnn.load_model(model_serialization_path, embeddings.lang_embeddings[emb_lang], loss_functions.softmax_cross_entropy,  just_predict = (labels is None))

	# preparing/cleaning the texts
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Preparing/cleaning the texts...', flush=True)
	texts_clean = [data_helper.clean_str(t.strip()).split() for t in texts]
	# encoding languages (full name to abbreviation)
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Encoding languages (full name to abbreviation)...', flush=True)
	langs = [map_lang(x) for x in languages]
	# preparing testing examples
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Preparing training examples...', flush=True)
	if labels:
		x_test, y_test, dist_labels = data_shaper.prep_classification(texts_clean, labels, embeddings, embeddings_language = emb_lang, multilingual_langs = langs, numbers_token = '<NUM/>', punct_token = '<PUNC/>', add_out_of_vocabulary_terms = False, dist_labels = cnn_classifier.dist_labels, max_seq_len = cnn_classifier.max_text_length)	
	else:	
		x_test = data_shaper.prep_classification(texts_clean, labels, embeddings, embeddings_language = emb_lang, multilingual_langs = langs, numbers_token = '<NUM/>', punct_token = '<PUNC/>', add_out_of_vocabulary_terms = False, dist_labels = cnn_classifier.dist_labels, max_seq_len = cnn_classifier.max_text_length)	
	
	simp_trainer = trainer.SimpleTrainer(cnn_classifier, session, build_feed_dict_func, None if not labels else eval_func, configuration_func = None)
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Starting test...', flush=True)
	results = simp_trainer.test(list(zip(x_test, y_test if labels else [None] * len(x_test))), parameters["batch_size"], eval_params = { "dist_labels" : cnn_classifier.dist_labels }, batch_size_irrelevant = True)
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Getting prediction labels...', flush=True)
	pred_labs = get_prediction_labels(results[0] if labels else results, cnn_classifier.dist_labels)
	
	if labels is None:
		io_helper.write_list(predictions_file_path, pred_labs)
	else:
		list_pairs = list(zip(pred_labs, labels))
		list_pairs.insert(0, ("Prediction", "Real label"))
		list_pairs.append(("Performance: ", str(results[1])))
		io_helper.write_list_tuples_separated(predictions_file_path, list_pairs)
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  Prediction is done!', flush=True)

def scale_efficient(filenames, texts, languages, embeddings, predictions_file_path, parameters, emb_lang = 'default', stopwords = []):
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Tokenizing documents...", flush = True)
	texts_tokenized = []
	for i in range(len(texts)):
		print("Document " + str(i + 1) + " of " + str(len(texts)), flush = True)
		texts_tokenized.append(simple_sts.simple_tokenize(texts[i], stopwords, lang_prefix = map_lang(languages[i])))

        
	embs_to_store = {filenames[x]: [texts_tokenized[x]] for x in range(len(texts_tokenized))} 
	print ("embs_to_store", len(embs_to_store))
    
	with open('tok-text.pickle', 'wb') as handle:
		pickle.dump(embs_to_store, handle, protocol=pickle.HIGHEST_PROTOCOL)	
	
        
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Building tf-idf indices for weighted aggregation...", flush = True)
	tf_index, idf_index = simple_sts.build_tf_idf_indices(texts_tokenized)
	agg_vecs = []
	for i in range(len(texts_tokenized)):
		print("Aggregating vector of the document: " + str(i+1) + " of " + str(len(texts_tokenized)), flush = True)
		#agg_vec = simple_sts.aggregate_weighted_text_embedding(embeddings, tf_index[i], idf_index, emb_lang, weigh_idf = (len(set(languages)) == 1))
		agg_vec = simple_sts.aggregate_weighted_text_embedding(embeddings, tf_index[i], idf_index, emb_lang, weigh_idf = False)
		agg_vecs.append(agg_vec)

        
        
	pairs = []
	cntr = 0
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Computing pairwise similarities...", flush = True)
	for i in range(len(agg_vecs) - 1):
		for j in range(i+1, len(agg_vecs)):
			cntr += 1
			#print("Pair: " + filenames[i] + " - " + filenames[j] + " (" + str(cntr) + " of " + str((len(filenames) * (len(filenames) - 1)) / 2))
			sim = 1.0 - spatial.distance.cosine(agg_vecs[i], agg_vecs[j])
			print (sim)
			#print("Similarity: " + str(sim))
			pairs.append((filenames[i], filenames[j], sim))

	# rescale distances and produce similarities
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Normalizing pairwise similarities...", flush = True)
	max_sim = max([x[2] for x in pairs])
	min_sim = min([x[2] for x in pairs])
	pairs = [(x[0], x[1], (x[2] - min_sim) / (max_sim - min_sim)) for x in pairs]

	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Fixing the pivot documents for scaling...", flush = True)
	min_sim_pair = [x for x in pairs if x[2] == 0][0]
	fixed = [(filenames.index(min_sim_pair[0]), -1.0), (filenames.index(min_sim_pair[1]), 1.0)]

	# propagating position scores, i.e., scaling
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Running graph-based label propagation with pivot rescaling and score normalization...", flush = True)
	g = graph.Graph(nodes = filenames, edges = pairs)
	scores = g.harmonic_function_label_propagation(fixed, rescale_extremes = True, normalize = True)
    
	embs_to_store = {filenames[x]: [agg_vecs[x],scores[filenames[x]]] for x in range(len(agg_vecs))} 
	print ("embs_to_store", len(embs_to_store))
    
	with open('docs-embs.pickle', 'wb') as handle:
		pickle.dump(embs_to_store, handle, protocol=pickle.HIGHEST_PROTOCOL)	
	
	if predictions_file_path:
		io_helper.write_dictionary(predictions_file_path, scores)
	return scores
		
def scale(filenames, texts, languages, embeddings, predictions_file_path, parameters, emb_lang = 'default'):
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"  Tokenizing documents...", flush=True)
	texts_tokenized = []
	for i in range(len(texts)):
		print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"  Document " + str(i + 1) + " of " + str(len(texts)), flush=True)
		texts_tokenized.append(simple_sts.simple_tokenize(texts[i], [], lang_prefix = map_lang(languages[i])))
	
	doc_dicts = []
	cntr = 0
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"  Building vocabularies for documents...", flush=True)
	for x in texts_tokenized:
		cntr += 1
		print("Document " + str(cntr) + " of " + str(len(texts)))
		doc_dicts.append(simple_sts.build_vocab(x, count_treshold = 1))

	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"  Computing similarities between document pairs...", flush=True)
	items = list(zip(filenames, languages, doc_dicts))
	pairs = []
	cntr = 0
	for i in range(len(items) - 1):
		for j in range(i+1, len(items)):
			cntr += 1
			print("Pair: " + items[i][0] + " - " + items[j][0] + " (" + str(cntr) + " of " + str((len(items) * (len(items) - 1)) / 2), flush=True)
			sim = simple_sts.greedy_alignment_similarity(embeddings, items[i][2], items[j][2], lowest_sim = 0.01, length_factor = 0.01)
			print("Similarity: " + str(sim), flush=True)
			print("\n", flush=True)
			pairs.append((items[i][0], items[j][0], sim))

	# rescale distances and produce similarities
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"  Normalizing pairwise similarities...", flush=True)
	max_sim = max([x[2] for x in pairs])
	min_sim = min([x[2] for x in pairs])
	pairs = [(x[0], x[1], (x[2] - min_sim) / (max_sim - min_sim)) for x in pairs]

	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"  Fixing the pivot documents for scaling...", flush=True)
	min_sim_pair = [x for x in pairs if x[2] == 0][0]
	fixed = [(filenames.index(min_sim_pair[0]), -1.0), (filenames.index(min_sim_pair[1]), 1.0)]

	# propagating position scores, i.e., scaling
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"  Running graph-based label propagation with pivot rescaling and score normalization...", flush=True)
	g = graph.Graph(nodes = filenames, edges = pairs)
	scores = g.harmonic_function_label_propagation(fixed, rescale_extremes = True, normalize = True)
	if predictions_file_path:
		io_helper.write_dictionary(predictions_file_path, scores)
	return scores

def topically_scale(filenames, texts, languages, embeddings, model_serialization_path, predictions_file_path, parameters, emb_lang = 'default', stopwords = []):
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Loading classifier...", flush=True)
	cnn_classifier, session = cnn.load_model(model_serialization_path, embeddings.lang_embeddings[emb_lang], loss_functions.softmax_cross_entropy,  just_predict = True)
	simp_trainer = trainer.SimpleTrainer(cnn_classifier, session, build_feed_dict_func, None, configuration_func = None)

	classified_texts = {}
	items = list(zip(filenames, texts, [map_lang(x) for x in languages]))
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Topically classifying texts...", flush = True)
	for item in items:
		fn, text, lang = item
		print(fn, flush=True)
		# split text in sentences
		sentences = nltk.sent_tokenize(text)
		sents_clean = [data_helper.clean_str(s.strip()).split() for s in sentences]
		langs = [lang] * len(sentences)
		
		# preparing training examples
		x_test = data_shaper.prep_classification(sents_clean, None, embeddings, embeddings_language = emb_lang, multilingual_langs = langs, numbers_token = '<NUM/>', punct_token = '<PUNC/>', add_out_of_vocabulary_terms = False, dist_labels = cnn_classifier.dist_labels, max_seq_len = cnn_classifier.max_text_length)
		
		results = simp_trainer.test(list(zip(x_test, [None]*len(x_test))), parameters["batch_size"], batch_size_irrelevant = True, print_batches = True)
		
		pred_labs = get_prediction_labels(results, cnn_classifier.dist_labels)
		print("Predictions: ", flush=True)
		print(pred_labs, flush=True)

		classified_texts[fn] = list(zip(sentences, pred_labs, langs))

		print("Languages: " + str(langs), flush=True)	
		print("Done with classifying: " + fn, flush=True)

	lines_to_write = []
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+ "  Topical scaling...", flush=True)
	for l in cnn_classifier.dist_labels:
		label_filtered = [(fn, classified_texts[fn][0][2], ' '.join([sent_label[0] for sent_label in classified_texts[fn] if sent_label[1] == l])) for fn in classified_texts]
		label_filtered = [x for x in label_filtered if len(x[2].strip()) > 50]
		if len(label_filtered) > 3:
			print("Topic: " + l, flush=True)
			fns = [x[0] for x in label_filtered]
			langs = [x[1] for x in label_filtered]
			filt_texts = [x[2] for x in label_filtered]
	
			for i in range(len(fns)):
				io_helper.write_list(os.path.dirname(predictions_file_path) + "/" + fns[i].split(".")[0] + "_" + l.replace(" ", "-") + ".txt", [filt_texts[i]])

			label_scale = scale_efficient(fns, filt_texts, [inverse_map_lang(x) for x in langs], embeddings, None, parameters, emb_lang = emb_lang, stopwords = stopwords)
			lines_to_write.append("Scaling for class: " + l)
			lines_to_write.extend([k + " " + str(label_scale[k]) for k in label_scale])
			lines_to_write.append("\n")
		else:
			lines_to_write.append("Topic: " + l + ": Insufficient number of files contains text of this topic (i.e., class) in order to allow for scaling for the topic.")
			print("Topic: " + l + ": Insufficient number of files contains text of this topic (i.e., class) in order to allow for scaling for the topic.", flush=True)
	
	io_helper.write_list(predictions_file_path, lines_to_write)
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'  Topical Scaling is done!', flush=True)


