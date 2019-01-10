import numpy as np
import gc
import os
import random
from collections import namedtuple

Example = namedtuple('Example', 'state action time')

class Dataset:

	def __init__(self, size, hist_len):
		self.size = size
		self.hist_len = hist_len
		self.states = np.empty((hist_len, size, 84, 84), dtype=np.float32)
		self.actions = np.empty(size, dtype=np.uint8)
		self.times = np.empty(size, dtype=np.uint8)
		self.index = 0
		self.sample_indices = range(size)
		self.shuffle_indices()
		self.minibatch_index = 0

	def shuffle_indices():
		random.shuffle(self.sample_indices)

	def clear(self):
		self.states = np.empty((self.size, 84, 84), dtype=np.uint8)
		self.actions = np.empty(self.size, dtype=np.uint8)
		self.times = np.empty(self.size, dtype=np.uint8)

	def store_experiences(self, storage_dir):
		np.save(os.path.join(storage_dir, "states"), self.states)
		np.save(os.path.join(storage_dir, "actions"), self.actions)
		np.save(os.path.join(storage_dir, "times"), self.times)

	def load_experiences(self, storage_dir):
		self.states = np.load(os.path.join(storage_dir, "states" + ".npy"))
		self.actions = np.load(os.path.join(storage_dir, "actions" + ".npy"))
		self.times = np.load(os.path.join(storage_dir, "times" + ".npy"))

	def add_item(self, state, action, time=None):
		if self.index == self.size:
			raise ValueError("Dataset is full. Clear dataset before adding anything.")
		# input a_t, r_t, f_t+1, episode done at t+1
		self.states[self.index, ...] = state
		self.actions[self.index] = action
		self.time[self.index] = time
		self.index += 1

	def sample_minibatch(self, batch_size):
		batch = []
		for _ in range(batch_size):
			index = self.sample_indices(self.minibatch_index)
			batch.append(Example(state=self.states[index],
								action=self.actions[index],
								time=self.time[index]))
			self.minibatch_index = self.minibatch_index + 1
			if self.minibatch_index >= self.size:
				self.minibatch_index = 0
				self.shuffle_indices()
		return batch