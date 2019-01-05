import numpy as np
import gc
import os

class Dataset:

	def __init__(self, size, hist_len):
		self.size = size
		self.hist_len = hist_len
		self.states = np.empty((size, 84, 84), dtype=np.uint8)
		self.actions = np.empty(size, dtype=np.uint8)
		self.times = np.empty(size dtype=np.uint8)
		self.index = 0

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
		# input a_t, r_t, f_t+1, episode done at t+1
		self.states[self.index, ...] = state
		self.actions[self.index] = action
		self.time[self.index] = reward
		self.terminals[self.index] = episode_done
		self.index += 1
		if self.index == STORAGE_BATCH_SIZE:
			self.store_experiences()
			self.reset()

	def get_next(self):
		if self.index == STORAGE_BATCH_SIZE:
			self.reset()
			self.load_experiences(self.count)
		state, action, reward, terminal = self.states[self.index], self.actions[self.index], self.rewards[self.index], self.terminals[self.index]
		self.index += 1
		return state, action, reward, terminal