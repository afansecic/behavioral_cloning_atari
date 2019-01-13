import os
from bc import Imitator
import numpy as np
from dataset import Example, Dataset
import utils
from ale_wrapper import ALEInterfaceWrapper

def train(rom,
		ale_seed,
		action_repeat_probability,
		learning_rate,
		alpha,
		min_squared_gradient,
		l2_penalty,
		minibatch_size, 
		hist_len,
		discount,
		checkpoint_dir,
		updates,
		dataset):


	ale = ALEInterfaceWrapper(action_repeat_probability)

	#Set the random seed for the ALE
	ale.setInt('random_seed', ALE_SEED)

	# Load the ROM file
	ale.loadROM(rom)

	print "Minimal Action set is:"
	print ale.getMinimalActionSet()

	# create DQN agent
	agent = Imitator(ale.getMinimalActionSet().tolist(),
				learning_rate,
				alpha,
				min_squared_gradient,
				checkpoint_dir,
				hist_len,
				l2_penalty)

	for update in range(updates):
		agent.train(dataset, 32)

if __name__ == '__main__':
	train()