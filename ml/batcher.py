import numpy as np
import random

def batch_iter(data, batch_size, num_epochs, shuffle = True):
		"""
		Generates a batch iterator for a dataset.
		"""
		#data = np.array(data, dtype = np.int32)
		data_size = len(data)

		num_batches_per_epoch = int(data_size/batch_size) + 1
		for epoch in range(num_epochs):
			# Shuffle the data at each epoch
			if shuffle:
				#shuffle_indices = np.random.permutation(np.arange(data_size))
				#shuffled_data = data[shuffle_indices]
				random.shuffle(data)
			#else:
			#	shuffled_data = data

			for batch_num in range(num_batches_per_epoch):
				start_index = batch_num * batch_size
				end_index = min((batch_num + 1) * batch_size, data_size)
				yield data[start_index:end_index]