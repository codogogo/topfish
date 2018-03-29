import numpy as np

def merge_confusion_matrices(conf_mats):
	res_mat = ConfusionMatrix(conf_mats[0].labels)
	for cm in conf_mats:
		res_mat.matrix = np.add(res_mat.matrix, cm.matrix)
	res_mat.compute_all_scores()
	return res_mat	

class ConfusionMatrix(object):
	"""
	Confusion matrix for evaluating classification tasks. 
	"""
	
	def __init__(self, labels = [], predictions = [], gold = [], one_hot_encoding = False, class_indices = False):
		# rows are true labels, columns predictions
		self.matrix = np.zeros(shape = (len(labels), len(labels)))
		self.labels = labels

		if len(predictions) != len(gold):
			raise ValueError("Predictions and gold labels do not have the same count.")
		for i in range(len(predictions)):
			index_pred = np.argmax(predictions[i]) if one_hot_encoding else (predictions[i] if class_indices else labels.index(predictions[i]))
			index_gold = np.argmax(gold[i]) if one_hot_encoding else (gold[i] if class_indices else labels.index(gold[i]))
			self.matrix[index_gold][index_pred] += 1

		if len(predictions) > 0: 
			self.compute_all_scores()

	def compute_all_scores(self):
		self.class_performances = {}
		self.counts = {}
		for i in range(len(self.labels)):
			tp = np.float32(self.matrix[i][i])
			fp_plus_tp = np.float32(np.sum(self.matrix, axis = 0)[i])
			fn_plus_tp = np.float32(np.sum(self.matrix, axis = 1)[i])
			p = tp / fp_plus_tp
			r = tp / fn_plus_tp
			self.class_performances[self.labels[i]] = (p, r, 2*p*r/(p+r))
			self.counts[self.labels[i]] = (tp, fp_plus_tp - tp, fn_plus_tp - tp)

		self.microf1 = np.float32(np.trace(self.matrix)) / np.sum(self.matrix)
		self.macrof1 = float(sum([x[2] for x in self.class_performances.values()])) / float(len(self.labels))
		self.macroP = float(sum([x[0] for x in self.class_performances.values()])) / float(len(self.labels))
		self.macroR = float(sum([x[1] for x in self.class_performances.values()])) / float(len(self.labels))
		self.accuracy = float(sum([self.matrix[i, i] for i in range(len(self.labels))])) / float(np.sum(self.matrix))
		

	def print_results(self):
		for l in self.labels:
			print(l + ": " + str(self.get_class_performance(l)))
		print("Micro avg: " + str(self.accuracy))
		print("Macro avg: " + str(self.macrof1))

	def get_class_performance(self, label):
		if label in self.labels:
			return self.class_performances[label]
		else:
			raise ValueException("Unknown label")

	def aggregate_class_performance(self, classes):
		true_sum = 0.0
		fp_sum = 0.0
		fn_sum = 0.0
		for l in classes:
			tp, fp, fn = self.counts[l]
			true_sum += tp
			fp_sum += fp
			fn_sum += fn
		p = true_sum / (fp_sum + true_sum)
		r = true_sum / (fn_sum + true_sum)
		f = (2 * r * p) / (r + p)
		return p, r, f
			
		

		
			 