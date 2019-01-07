import argparse
from offline_storage import StorageBuffer
from pdb import set_trace
import os
import numpy as np
import replaybuffer

class Episode(object):

	def __init__(self, start, end, storage_size, offline_dir, hist_len):
		assert end > start
		self.start = start
		self.end = end
		self.storage_size = storage_size
		self.offline_dir = offline_dir

		num_items = end - start + 1
		self.episode_len = num_items - hist_len
		self.states = np.empty((num_items, 84, 84), dtype=np.uint8)
		self.actions = np.empty(num_items, dtype=np.uint8)
		self.rewards = np.empty(num_items)
		self.terminals = np.empty(num_items, dtype=np.bool)
		self.hist_len = hist_len

		buffers = self.get_episode(self.start, self.end, self.storage_size, self.offline_dir)
		start_index = start % self.storage_size
		index = start_index
		storage_index = 0
		storage_buffer = buffers[storage_index]
		episode_index = 0
		for i in range(num_items):
			if index > self.storage_size:
				index = 0
				storage_index += 1
				storage_buffer = buffers[storage_index]
			self.states[episode_index] = storage_buffer[index][0]
			self.actions[episode_index] = storage_buffer[index][1]
			self.rewards[episode_index] = storage_buffer[index][2]
			self.terminals[episode_index] = storage_buffer[index][3]
			episode_index += 1
			index += 1
		self.total_reward = np.sum(self.rewards)

	'''
	Assumes indices chosen properly
	'''
	def get_state(self, index):
		return self.states[[(index - i) for i in reversed(range(self.hist_len))],...]

	#assumes 0 indexing
	def get_episode(self, start, end, storage_size, offline_dir):
		batch_num_start = start/storage_size
		batch_num_end = end/storage_size
		storage_buffs = [StorageBuffer(self.storage_size) for i in range(batch_num_start, batch_num_end + 1)]
		for i in range(batch_num_start, batch_num_end + 1):
			storage_buffs[i].load_experiences(offline_dir, i)
		return storage_buffs

	def __getitem__(self, index):
		assert 0 <= index < self.episode_len
		index = index + self.hist_len
		state = np.expand_dims(self.get_state(index - 1).astype(np.float32)/255.0, axis=0)
		action = self.actions[index]
		reward = self.rewards[index]
		next_state = np.expand_dims(self.get_state(index).astype(np.float32)/255.0, axis=0)
		terminal = self.terminals[index]
		return {'state':state, 'action':action, 
				'reward':reward, 'next_state':next_state, 'terminal':terminal}


def extract_episodes(storage_dir, storage_size):
	episodes = []
	save_number = 0
	timestep = 0
	start = 0
	reward = 0
	while os.path.exists(os.path.join(storage_dir, "states" + str(save_number) + ".npy")):
		rewards = np.load(os.path.join(storage_dir, "rewards" + str(save_number) + ".npy"))
		terminals = np.load(os.path.join(storage_dir, "terminals" + str(save_number) + ".npy"))	
		for _ in range(storage_size):
			reward += rewards[timestep % storage_size]
			if terminals[timestep % storage_size]:
				episodes.append((start, timestep, reward))
				start = timestep + 1
				reward = 0
			timestep +=1
		save_number += 1
	return episodes
