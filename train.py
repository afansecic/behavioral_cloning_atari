import os
from bc import Imitator
import numpy as np
from dataset import Example, Dataset
import utils
from ale_wrapper import ALEInterfaceWrapper
from evaluator import Evaluator
from pdb import set_trace
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#try bmh
plt.style.use('bmh')


def smooth(losses, run=10):
	new_losses = []
	for i in range(len(losses)):
		new_losses.append(np.mean(losses[max(0, i - 10):i+1]))
	return new_losses

def plot(losses):
		p=plt.plot(smooth(losses, 25))
		plt.xlabel("Update")
		plt.ylabel("Loss")
		plt.legend(loc='lower center')
		plt.savefig('/u/prabhatn/behavioral_cloning_atari/checkpoints/loss.png')

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
	log_frequency = 1000
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
	#Plot losses
	losses = []
	for loss in agent.losses:
		losses.append(loss.data.cpu().numpy())
	plot(losses)
	#Evaluation
	evaluator = Evaluator(rom=rom)
	evaluator.evaluate(agent)


if __name__ == '__main__':
	train()