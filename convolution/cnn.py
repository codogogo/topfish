import tensorflow as tf
import numpy as np
from helpers import io_helper

def load_labels_and_max_length(path):
	parameters, model = io_helper.deserialize(path)
	return parameters["dist_labels"], parameters["max_text_length"]

def load_model(path, embeddings, loss_function, just_predict = True):
	parameters, model = io_helper.deserialize(path)

	print("Defining and initializing model...")
	classifier = CNN(embeddings = (parameters["embedding_size"], embeddings), num_conv_layers = parameters["num_convolutions"], filters = parameters["filters"], k_max_pools = parameters["k_max_pools"], manual_features_size = parameters["manual_features_size"])
	classifier.define_model(parameters["max_text_length"], parameters["num_classes"], loss_function, -1, l2_reg_factor = parameters["reg_factor"], update_embeddings = parameters["upd_embs"])
	if not just_predict:
		classifier.define_optimization(learning_rate = parameters["learning_rate"])

	print("Initializing session...", flush = True)
	session = tf.InteractiveSession()
	session.run(tf.global_variables_initializer())

	classifier.set_variable_values(session, model)
	classifier.set_distinct_labels(parameters["dist_labels"])

	return classifier, session

class CNN(object):
	"""
	A general convolutional neural network for text classification.
	The CNN is highly customizable, the user may determine the number of convolutional and pooling layers and all other parameters of the network (e.g., the number of filters and filter sizes)  
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	"""

	def __init__(self, embeddings = (100, None), num_conv_layers = 1, filters = [[(3, 64), (4, 128), (5, 64)]], k_max_pools = [1], manual_features_size = 0):
		self.emb_size = embeddings[0]
		self.embs = embeddings[1]
		self.num_convolutions = num_conv_layers
		self.filters = filters
		self.k_max_pools = k_max_pools

		self.variable_memory = {}
		self.manual_features_size = manual_features_size

	def define_model(self, max_text_length, num_classes, loss_function, vocab_size, l2_reg_factor = 0.0, update_embeddings = False):
		self.update_embeddings = update_embeddings
		self.reg_factor = l2_reg_factor
		self.max_text_length = max_text_length
		self.num_classes = num_classes
		self.loss_function = loss_function

		self.input_x = tf.placeholder(tf.int32, [None, max_text_length], name="input_x")
		if self.manual_features_size > 0:
			self.manual_features = tf.placeholder(tf.float32, [None, self.manual_features_size], name="man_feats")
		self.dropout = tf.placeholder(tf.float32, name="dropout")

		if self.embs is None:
			self.W_embeddings = tf.Variable(tf.random_uniform([vocab_size, self.emb_size], -1.0, 1.0), name="W_embeddings")
		elif update_embeddings:
			self.W_embeddings = tf.Variable(self.embs, dtype = tf.float32, name="W_embeddings")
		else:
			self.W_embeddings = tf.constant(self.embs, dtype = tf.float32, name="W_embeddings")

		self.mb_embeddings = tf.expand_dims(tf.nn.embedding_lookup(self.W_embeddings, self.input_x), -1)

		for i in range(self.num_convolutions):
			current_filters = self.filters[i]
			current_max_pool_size = self.k_max_pools[i]

			if i > 0:
				pooled = tf.reshape(pooled, [-1, self.k_max_pools[i - 1], sum_filt, 1]) 

			input = self.mb_embeddings if i == 0 else pooled
			input_dim = self.emb_size if i == 0 else sum_filt
			num_units = max_text_length if i == 0 else self.k_max_pools[i - 1]

			sum_filt = 0
			for filter_size, num_filters in current_filters:				
				filter_shape = [filter_size, input_dim, 1, num_filters]
				
				W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, dtype = tf.float32), name="W_conv_" + str(i) + "_" + str(filter_size))
				self.variable_memory["W_conv_" + str(i) + "_" + str(filter_size)] = W_conv

				b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype = tf.float32), name="b_" + str(i) + "_" + str(filter_size))
				self.variable_memory["b_" + str(i) + "_" + str(filter_size)] = b_conv

				conv = tf.nn.conv2d(input, W_conv, strides=[1, 1, 1, 1], padding="VALID", name="conv_" + str(i) + "_" + str(filter_size))
				h = tf.nn.relu(tf.nn.bias_add(conv, b_conv), name="relu" + str(i) + "_" + str(filter_size))

				if sum_filt == 0:
					pooled = tf.nn.max_pool(h, ksize=[1, (num_units - filter_size + 1) - current_max_pool_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool_" + str(i) + "_" + str(filter_size))
					
				else:
					new_pool = tf.nn.max_pool(h, ksize=[1, (num_units - filter_size + 1) - current_max_pool_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool_" + str(i) + "_" + str(filter_size))					
					pooled = tf.concat(axis=3, values=[pooled, new_pool])

				sum_filt += num_filters
		
		self.pooled_flat = tf.reshape(pooled, [-1, self.k_max_pools[-1] * sum_filt])
		self.pooled_dropout = tf.nn.dropout(self.pooled_flat, self.dropout)

		W_softmax = tf.get_variable("W_softmax", shape=[self.k_max_pools[-1] * sum_filt + self.manual_features_size, num_classes], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
		self.variable_memory["W_softmax"] = W_softmax

		b_softmax = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype = tf.float32), name="b_softmax")
		self.variable_memory["b_softmax"] = b_softmax

		self.final_features = tf.concat(axis=1, values=[self.pooled_dropout, self.manual_features]) if self.manual_features_size > 0 else self.pooled_dropout  
		self.preds = tf.nn.xw_plus_b(self.final_features, W_softmax, b_softmax, name="scores")
		#self.preds_sftmx = tf.nn.softmax(self.preds)

		self.l2_loss = tf.constant(0.0)
		self.l2_loss += tf.nn.l2_loss(W_softmax)
		self.l2_loss += tf.nn.l2_loss(b_softmax)

	
	def define_optimization(self, learning_rate = 1e-3):
		self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
		self.pure_loss = self.loss_function(self.preds, self.input_y)
		self.loss = self.pure_loss + self.reg_factor * self.l2_loss

		self.learning_rate = learning_rate
		self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

	def set_distinct_labels(self, dist_labels):
		self.dist_labels = dist_labels
		
	def get_feed_dict(self, input_data, labels, dropout, manual_feats = None):
		fd_mine = { self.input_x : input_data, self.dropout : dropout }
		if labels is not None:
			fd_mine.update({self.input_y : labels})
		if manual_feats is not None:
			fd_mine.update({self.manual_features : manual_feats})
		return fd_mine

	def get_variable_values(self, session):
		variables = {}
		for v in self.variable_memory:
			value = self.variable_memory[v].eval(session = session)
			variables[v] = value	
		return variables
	
	def set_variable_values(self, session, var_values):
		for v in var_values:
			session.run(self.variable_memory[v].assign(var_values[v]))

	def get_hyperparameters(self):
		params = { "embedding_size" : self.emb_size,
				  "num_convolutions" : self.num_convolutions,
				  "filters" : self.filters, 
				  "k_max_pools" : self.k_max_pools, 
				  "upd_embs" : self.update_embeddings, 
				  "reg_factor" : self.reg_factor, 
				  "learning_rate" : self.learning_rate, 
				  "manual_features_size" : self.manual_features_size, 
				  "max_text_length" : self.max_text_length,
				  "num_classes" : self.num_classes, 
				  "dist_labels" : self.dist_labels }
		return params

	def get_model(self, session):
		return [self.get_hyperparameters(), self.get_variable_values(session)]

	def serialize(self, session, path):
		variables = self.get_variable_values(session)
		to_serialize = [self.get_hyperparameters(), self.get_variable_values(session)]
		io_helper.serialize(to_serialize, path)		
			
			
			
				
	