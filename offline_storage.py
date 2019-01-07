import numpy as np
import gc
import os

class StorageBuffer(object):

	def __init__(self, storage_size):
		self.storage_size = storage_size
		self.states = np.empty((storage_size, 84, 84), dtype=np.uint8)
		self.actions = np.empty(storage_size, dtype=np.uint8)
		self.rewards = np.empty(storage_size)
		self.terminals = np.empty(storage_size, dtype=np.bool)

	def load_experiences(self, offline_dir, save_number):
		self.states = np.load(os.path.join(offline_dir, "states" + str(save_number) + ".npy"))
		self.actions = np.load(os.path.join(offline_dir, "actions" + str(save_number) + ".npy"))
		self.rewards = np.load(os.path.join(offline_dir, "rewards" + str(save_number) + ".npy"))
		self.terminals = np.load(os.path.join(offline_dir, "terminals" + str(save_number) + ".npy"))

	def __getitem__(self, i):
		assert 0 <= i < self.storage_size
		state = self.states[i]
		action = self.actions[i]
		reward = self.rewards[i]
		terminal = self.terminals[i]
		return state, action, reward, terminal


class OfflineStorage:

	def __init__(self, storage_batch_size, offline_dir):
		self.storage_batch_size = storage_batch_size
		self.states = np.empty((storage_batch_size, 84, 84), dtype=np.uint8)
		self.actions = np.empty(storage_batch_size, dtype=np.uint8)
		self.rewards = np.empty(storage_batch_size)
		self.terminals = np.empty(storage_batch_size, dtype=np.bool)
		self.offline_dir = os.path.join(offline_dir,"")
		self.index = 0
		self.count = 0

	def reset(self):
		self.states = np.empty((self.storage_batch_size, 84, 84), dtype=np.uint8)
		self.actions = np.empty(self.storage_batch_size, dtype=np.uint8)
		self.rewards = np.empty(self.storage_batch_size)
		self.terminals = np.empty(self.storage_batch_size, dtype=np.bool)
		self.index = 0
		gc.collect()

	def store_experiences(self):
		np.save(os.path.join(self.offline_dir, "states" + str(self.count)), self.states)
		np.save(os.path.join(self.offline_dir, "actions" + str(self.count)), self.actions)
		np.save(os.path.join(self.offline_dir, "rewards" + str(self.count)), self.rewards)
		np.save(os.path.join(self.offline_dir, "terminals" + str(self.count)), self.terminals)
		self.count += 1

	def load_experiences(self, save_number):
		self.states = np.load(os.path.join(self.offline_dir, "states" + str(save_number) + ".npy"))
		self.actions = np.load(os.path.join(self.offline_dir, "actions" + str(save_number) + ".npy"))
		self.rewards = np.load(os.path.join(self.offline_dir, "rewards" + str(save_number) + ".npy"))
		self.terminals = np.load(os.path.join(self.offline_dir, "terminals" + str(save_number) + ".npy"))
		self.count += 1

	def add_item(self, frame, action, reward, episode_done):
		# input a_t, r_t, f_t+1, episode done at t+1
		self.states[self.index, ...] = frame
		self.actions[self.index] = action
		self.rewards[self.index] = reward
		self.terminals[self.index] = episode_done
		self.index += 1
		if self.index == self.storage_batch_size:
			self.store_experiences()
			self.reset()

	def get_next(self):
		if self.index == self.storage_batch_size:
			self.reset()
			self.load_experiences(self.count)
		state, action, reward, terminal = self.states[self.index], self.actions[self.index], self.rewards[self.index], self.terminals[self.index]
		self.index += 1
		return state, action, reward, terminal


