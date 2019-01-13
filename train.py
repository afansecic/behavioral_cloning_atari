import os
from bc import Imitator
import numpy as np
from dataset import Example, Dataset
import utils
from ale_wrapper import ALEInterfaceWrapper
from evaluator import Evaluator

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
	ale.setInt('random_seed', ale_seed)

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

	print "Beginning training..."
	log_frequency = 10
	log_num = log_frequency
	update = 1
	while update < updates:
		if update > log_num:
			print(str(update) + " updates completed.")
			log_num += log_frequency
		agent.train(dataset, 32)
		update += 1
	print "Training completed."
	agent.checkpoint_network()

	#Evaluation
	evaluator = Evaluator(rom=rom)
	evaluator.evaluate(agent)


if __name__ == '__main__':
	train()