import numpy as np
from evaluation import confusion_matrix
from ml import batcher
import random
import copy
import tensorflow as tf
from sys import stdin

class SimpleTrainer(object):
	def __init__(self, model, session, feed_dict_function, eval_func, configuration_func = None, labels = None, additional_results_func = None):
		self.model = model
		self.session = session
		self.feed_dict_function = feed_dict_function
		self.eval_func = eval_func
		self.config_func = configuration_func
		self.additional_results_function = additional_results_func
		self.labels = labels

	def train_model_single_iteration(self, feed_dict):
		self.model.train_step.run(session = self.session, feed_dict = feed_dict)

	def predict(self, feed_dict):
		return self.model.preds.eval(session = self.session, feed_dict = feed_dict)
	
	def evaluate(self, feed_dict, gold):
		preds = predict(self.model, self.session, feed_dict)
		return preds, self.eval_func(gold, preds)

	def test(self, test_data, batch_size, eval_params = None, print_batches = False, batch_size_irrelevant = True, compute_loss = False):
		if compute_loss:
			epoch_loss = 0
		batches_eval = batcher.batch_iter(test_data, batch_size, 1, shuffle = False)
		eval_batch_counter = 1
				
		for batch_eval in batches_eval:
			if (batch_size_irrelevant or len(batch_eval) == batch_size):
				feed_dict_eval, golds_batch_eval = self.feed_dict_function(self.model, batch_eval, None, predict = True)	
				preds_batch_eval = self.predict(feed_dict_eval)
				if compute_loss:
					batch_eval_loss = self.model.loss.eval(session = self.session, feed_dict = feed_dict_eval)
					epoch_loss += batch_eval_loss

				if eval_batch_counter == 1:
					golds = golds_batch_eval
					preds = preds_batch_eval
				else:
					golds = np.concatenate((golds, golds_batch_eval), axis = 0)
					preds = np.concatenate((preds, preds_batch_eval), axis = 0)
				if print_batches:
					print("Eval batch counter: " + str(eval_batch_counter), flush=True)
			eval_batch_counter += 1

		if self.eval_func is not None:
			score = self.eval_func(golds, preds, eval_params)
			if compute_loss:
				return preds, score, epoch_loss 
			else:
				return preds, score
		else:
			if compute_loss:
				return preds, epoch_loss
			else:
				return preds 

	def train(self, train_data, batch_size, max_num_epochs, num_epochs_not_better_end = 5, epoch_diff_smaller_end = 1e-5, print_batch_losses = True, configuration = None, eval_params = None, shuffle_data = True, batch_size_irrelevant = True):
		batch_counter = 0
		epoch_counter = 0
		epoch_losses = []
		epoch_loss = 0
		batches_in_epoch = int(len(train_data)/batch_size) + 1

		batches = batcher.batch_iter(train_data, batch_size, max_num_epochs, shuffle = shuffle_data)
		for batch in batches:
			batch_counter += 1

			if (batch_size_irrelevant or len(batch) == batch_size):
				feed_dict, gold_labels = self.feed_dict_function(self.model, batch, config = configuration, predict = False)
				self.train_model_single_iteration(feed_dict)
				batch_loss = self.model.loss.eval(session = self.session, feed_dict = feed_dict)
				if print_batch_losses:
					print("Batch " + str(batch_counter) + ": " + str(batch_loss), flush=True)

			if batch_counter % batches_in_epoch == 0:
				epoch_counter += 1
				print("Evaluating the epoch loss for epoch " + str(epoch_counter), flush=True)
				
				if self.eval_func: 
					preds, score, epoch_loss = self.test(train_data, batch_size, eval_params, False, batch_size_irrelevant = batch_size_irrelevant, compute_loss = True)
				else: 
					preds, epoch_loss = self.test(train_data, batch_size, None, False, batch_size_irrelevant = batch_size_irrelevant, compute_loss = True)

				print("Epoch " + str(epoch_counter) + ": " + str(epoch_loss), flush=True)
				if self.eval_func: 
					print("Epoch (train) performance: " + str(score), flush=True)
				print("Previous epochs: " + str(epoch_losses), flush=True)

				if len(epoch_losses) == num_epochs_not_better_end and (epoch_losses[0] - epoch_loss < epoch_diff_smaller_end):
					break
				else: 
					epoch_losses.append(epoch_loss)
					epoch_loss = 0
					if len(epoch_losses) > num_epochs_not_better_end:
						epoch_losses.pop(0)

	def train_dev(self, train_data, dev_data, batch_size, max_num_epochs, num_devs_not_better_end = 5, batch_dev_perf = 100, print_batch_losses = True, dev_score_maximize = True, configuration = None, print_training = False, shuffle_data = True):
		batch_counter = 0
		epoch_counter = 0
		epoch_losses = []
		dev_performances = []
		dev_losses = []
		epoch_loss = 0
		
		best_model = None
		best_performance = -1
		best_preds_dev = None	
		batches_in_epoch = int(len(train_data)/batch_size) + 1

		batches = batcher.batch_iter(train_data, batch_size, max_num_epochs, shuffle = shuffle_data)
		for batch in batches:
			batch_counter += 1

			if (len(batch) == batch_size):
				feed_dict, gold_labels = self.feed_dict_function(self.model, batch, configuration)
				self.train_model_single_iteration(feed_dict)
			
				batch_loss = self.model.pure_loss.eval(session = self.session, feed_dict = feed_dict)
				#batch_dist_loss = self.model.distance_loss.eval(session = self.session, feed_dict = feed_dict) 
				epoch_loss += batch_loss

				if print_training and print_batch_losses:
					print("Batch loss" + str(batch_counter) + ": " + str(batch_loss), flush=True)
					#print("Batch distance loss" + str(batch_counter) + ": " + str(batch_dist_loss))

			if batch_counter % batches_in_epoch == 0:
				epoch_counter += 1
				if print_training: 
					print("\nEpoch " + str(epoch_counter) + ": " + str(epoch_loss), flush=True)
					print("Previous epochs: " + str(epoch_losses) + "\n", flush=True)
				epoch_losses.append(epoch_loss)
				epoch_loss = 0
				if len(epoch_losses) > num_devs_not_better_end:
					epoch_losses.pop(0)
		
			if batch_counter % batch_dev_perf == 0:
				if print_training:
					print("\n### Evaluation of development set, after batch " + str(batch_counter) + " ###", flush=True)
				batches_dev = batcher.batch_iter(dev_data, batch_size, 1, shuffle = False)
				dev_batch_counter = 1
				dev_loss = 0
				for batch_dev in batches_dev:
					if (len(batch_dev) == batch_size):
						feed_dict_dev, golds_batch_dev = self.feed_dict_function(self.model, batch_dev, configuration, predict = True)	
						dev_batch_loss = self.model.pure_loss.eval(session = self.session, feed_dict = feed_dict_dev)
						dev_loss += dev_batch_loss
						if print_training and print_batch_losses: 
							print("Dev batch: " + str(dev_batch_counter) + ": " + str(dev_batch_loss), flush=True)
						preds_batch_dev = self.predict(feed_dict_dev) 
						if dev_batch_counter == 1:
							golds = golds_batch_dev
							preds = preds_batch_dev
						else:
							golds = np.concatenate((golds, golds_batch_dev), axis = 0)
							preds = np.concatenate((preds, preds_batch_dev), axis = 0)
					dev_batch_counter += 1
				print("Development pure loss: " + str(dev_loss), flush=True)
				score = self.eval_func(golds, preds, self.labels)
				if self.additional_results_function:
					self.additional_results_function(self.model, self.session)
				if print_training:
					print("Peformance: " + str(score) + "\n", flush=True)
					print("Previous performances: " + str(dev_performances), flush=True)
					print("\nLoss: " + str(dev_loss) + "\n", flush=True)
					print("Previous losses: " + str(dev_losses), flush=True)
				if score > best_performance:
					best_model = self.model.get_model(self.session)
					best_preds_dev = preds
					best_performance = score
			
				#if len(dev_performances) == num_devs_not_better_end and ((dev_score_maximize and dev_performances[0] >= score) or (not dev_score_maximize and dev_performances[0] <= score)):
				if len(dev_losses) == num_devs_not_better_end and dev_losses[0] < dev_loss:
					break
				else: 
					dev_performances.append(score)
					dev_losses.append(dev_loss)
					if len(dev_performances) > num_devs_not_better_end:
						dev_performances.pop(0)
						dev_losses.pop(0) 			
		return (best_model, best_performance, best_preds_dev, golds)

	def cross_validate(self, data, batch_size, max_num_epochs, num_folds = 5, num_devs_not_better_end = 5, batch_dev_perf = 100, print_batch_losses = True, dev_score_maximize = True, configuration = None, print_training = False, micro_performance = True, shuffle_data = True):
		folds = np.array_split(data, num_folds)
		results = {}

		for i in range(num_folds):
			train_data = []
			for j in range(num_folds):
				if j != i:
					train_data.extend(folds[j])
			dev_data = folds[i]

			print("Sizes: train " + str(len(train_data)) + "; dev " + str(len(dev_data)), flush=True)
			print("Fold " + str(i+1) + ", creating model...", flush=True)
			model, conf_str, session = self.config_func(configuration)
			self.model = model
			self.session = session
			print("Fold " + str(i+1) + ", training the model...", flush=True)
			results[conf_str + "__fold-" + str(i+1)] = self.train_dev(train_data, dev_data, batch_size, max_num_epochs, num_devs_not_better_end, batch_dev_perf, print_batch_losses, dev_score_maximize, configuration, print_training, shuffle_data = shuffle_data)
			
			print("Closing session, reseting the default graph (freeing memory)", flush=True)
			self.session.close()
			tf.reset_default_graph()
			print("Performance: " + str(results[conf_str + "__fold-" + str(i+1)][1]), flush=True)
		
		if micro_performance:
			print("Concatenating fold predictions for micro-performance computation", flush=True)
			cntr = 0
			for k in results:
				cntr += 1
				if cntr == 1:
					all_preds = results[k][2]
					all_golds = results[k][3]
				else:
					all_preds = np.concatenate((all_preds, results[k][2]), axis = 0)
					all_golds = np.concatenate((all_golds, results[k][3]), axis = 0)	
			micro_perf = self.eval_func(all_golds, all_preds, self.labels)
			return results, micro_perf
		else: 
			return results	

	def grid_search(self, configurations, train_data, dev_data, batch_size, max_num_epochs, num_devs_not_better_end = 5, batch_dev_perf = 100, print_batch_losses = True, dev_score_maximize = True, cross_validate = False, cv_folds = None, print_training = False, micro_performance = False, shuffle_data = True):
		if self.config_func is None:
			raise ValueError("Function that creates a concrete model for a given hyperparameter configuration must be defined!")
		results = {}
		config_cnt = 0
		for config in configurations:
			config_cnt += 1
			print("Config: #" + str(config_cnt), flush=True)
			if cross_validate:
				results[str(config)] = self.cross_validate(train_data, batch_size, max_num_epochs, cv_folds, num_devs_not_better_end, batch_dev_perf, print_batch_losses, dev_score_maximize, config, print_training, micro_performance = micro_performance, shuffle_data = shuffle_data)
				if micro_performance:
					print("### Configuration performance: " + str(results[str(config)][1]), flush=True)
			else:
				model, conf_str, session = self.config_func(config)
				self.model = model
				self.session = session
				results[conf_str] = self.train_dev(train_data, dev_data, batch_size, max_num_epochs, num_devs_not_better_end, batch_dev_perf, print_batch_losses, dev_score_maximize, config, print_training, shuffle_data = shuffle_data)
				
				print("Closing session, reseting the default graph (freeing memory)", flush=True)
				self.session.close()
				tf.reset_default_graph()
		return results
			

class Trainer(object):
	"""
	A wrapper around the classifiers, implementing functionality like cross-validation, batching, grid search, etc.
	"""
	def __init__(self, classifier, one_hot_encoding_preds = False, class_indexes = True):
		self.classifier = classifier
		self.one_hot_encoding_preds = one_hot_encoding_preds
		self.class_indices = class_indexes
	
	def cross_validate(self, tf_session, class_labels, data_input, data_labels, num_folds, batch_size, num_epochs, model_reset_function = None, shuffle = False, fold_avg = 'micro', cl_perf = None, overall_perf = True, num_epochs_not_better_end = 2):
		conf_matrices = []
		best_epochs = []	
		if shuffle:
			paired = list(zip(data_input, data_labels))
			random.shuffle(paired)	
			data_input, data_labels = zip(*paired)

		folds = self.cross_validation_fold(data_input, data_labels, num_folds)
		fold_counter = 1
		for fold in folds:
			print("Fold: " + str(fold_counter), flush=True)
			train_input = fold[0]; train_labels = fold[1]; dev_input = fold[2]; dev_labels = fold[3]
			model_reset_function(tf_session)
			conf_mat, epoch = self.train_and_test(tf_session, class_labels, train_input, train_labels, dev_input, dev_labels, batch_size, num_epochs, cl_perf, overall_perf, num_epochs_not_better_end = num_epochs_not_better_end)
			conf_matrices.append(conf_mat)
			best_epochs.append(epoch)
			fold_counter += 1
		if fold_avg == 'macro':
			return conf_matrices, best_epochs
		elif fold_avg == 'micro':
			return confusion_matrix.merge_confusion_matrices(conf_matrices), best_epochs
		else:
			raise ValueError("Unknown value for fold_avg")	

		
	def cross_validation_fold(self, data_input, data_labels, num_folds):
		folds_x_train = np.array_split(data_input, num_folds)
		folds_y_train = np.array_split(data_labels, num_folds)
		for i in range(num_folds):
			train_set_x = []
			train_set_y = []	
			for j in range(num_folds):
				if j != i:
					train_set_x.extend(folds_x_train[j])
					train_set_y.extend(folds_y_train[j])
			dev_set_x = folds_x_train[i]
			dev_set_y = folds_y_train[i]
			yield [np.array(train_set_x), np.array(train_set_y), dev_set_x, dev_set_y]

	def train_and_test(self, session, class_labels, x_train, y_train, x_test, y_test, batch_size, num_epochs, cl_perf = None, overall_perf = True, num_epochs_not_better_end = 10, manual_features = False):
		batch_counter = 0
		epoch_loss = 0
		epoch_counter = 0
		last_epoch_results = []
		best_f = 0
		best_epoch = 0
		best_conf_mat = None
		best_predictions = []

		num_batches_per_epoch = int((len(x_train) if not manual_features else len(x_train[0])) / batch_size) + 1

		batches = batcher.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs) if not manual_features else batcher.batch_iter(list(zip(x_train[0], x_train[1], y_train)), batch_size, num_epochs)
		for batch in batches:
			if manual_features:
				x_b, x_b_man, y_b = zip(*batch)	
				batch_loss = self.classifier.train(session, x_b, y_b, man_feats = x_b_man)
			else:
				x_b, y_b = zip(*batch)
				x_b = np.array(x_b)
				y_b = np.array(y_b)
				batch_loss = self.classifier.train(session, x_b, y_b)
			epoch_loss += batch_loss

			batch_counter += 1

			#if batch_counter % 50 == 0:
				#print("Batch " + str(batch_counter) + " loss: " + str(batch_loss))
				# evaluating current model's performance on test
				#preds, gold = self.classifier.predict(session, x_test, y_test)
				#self.evaluate_performance(class_labels, preds, gold, cl_perf, overall_perf, " (test set) ")

			if batch_counter % num_batches_per_epoch == 0:
				epoch_counter += 1
				print("Epoch " + str(epoch_counter) + " loss: " + str(epoch_loss), flush=True)
				last_epoch_results.append(epoch_loss)
				epoch_loss = 0

				if manual_features:
					x_test_text = x_test[0]
					x_test_manual = x_test[1]
					preds, gold = self.classifier.predict(session, x_test_text, y_test, man_feats = x_test_manual)	

				else: 
					preds, gold = self.classifier.predict(session, x_test, y_test)

				cm = self.evaluate_performance(class_labels, preds, gold, cl_perf, overall_perf, " (test set) ")
				
				fepoch = cm.accuracy # cm.get_class_performance("1")[2]
				if fepoch > best_f:
					best_f = fepoch
					best_epoch = epoch_counter
					best_conf_mat = cm
					best_predictions = preds
		
				if len(last_epoch_results) > num_epochs_not_better_end:
					last_epoch_results.pop(0)
				print("Last epochs: " + str(last_epoch_results), flush=True)

				if len(last_epoch_results) == num_epochs_not_better_end and last_epoch_results[0] < last_epoch_results[-1]:
					print("End condition satisfied, training finished. ", flush=True)
					break

		#preds, gold = self.classifier.predict(session, x_train, y_train)
		#self.evaluate_performance(class_labels, preds, gold, cl_perf, overall_perf, " (train set) ")

		#preds, gold = self.classifier.predict(session, x_test, y_test)
		#conf_mat = self.evaluate_performance(class_labels, preds, gold, cl_perf, overall_perf, " (test set) ")
		#return conf_mat
		return best_conf_mat, best_epoch, best_predictions
			
	def evaluate_performance(self, class_labels, preds, gold, cl_perf = None, overall_perf = True, desc = " () ", print_perf = True):
		conf_matrix = confusion_matrix.ConfusionMatrix(class_labels, preds, gold, self.one_hot_encoding_preds, self.class_indices)
		if print_perf:
			if cl_perf is not None:
				for cl in cl_perf:
					p, r, f = conf_matrix.get_class_performance(cl)
					print(desc + " Class: " + cl + "\nP: " + str(p) + "\nR: " + str(r) + "\nF: " + str(f) + "\n", flush=True)
			if overall_perf:
				print(desc + " Micro F1: " + str(conf_matrix.microf1) + "\nMacro F1: " + str(conf_matrix.macrof1) + "\n", flush=True)
		return conf_matrix