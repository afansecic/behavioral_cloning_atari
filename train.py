import os
from bc import Imitator
import numpy as np
from dataset import Example, Dataset
import utils
#from ale_wrapper import ALEInterfaceWrapper
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

def plot(losses, checkpoint_dir, env_name):
		print("Plotting losses to ", os.path.join(checkpoint_dir, env_name + "_loss.png"))
		p=plt.plot(smooth(losses, 25))
		plt.xlabel("Update")
		plt.ylabel("Loss")
		plt.legend(loc='lower center')
		plt.savefig(os.path.join(checkpoint_dir, env_name + "loss.png"))

def train(env_name,
		minimal_action_set,
		learning_rate,
		alpha,
		l2_penalty,
		minibatch_size,
		hist_len,
		discount,
		checkpoint_dir,
		updates,
		dataset,
		num_eval_episodes,
		epsilon_greedy,
		extra_info):

	# create DQN agent
	agent = Imitator(list(minimal_action_set),
				learning_rate,
				alpha,
				checkpoint_dir,
				hist_len,
				l2_penalty)

	print("Beginning training...")
	log_frequency = 1000
	log_num = log_frequency
	update = 1
	running_loss = 0.
	while update < updates:
		if update > log_num:
			print(str(update) + " updates completed. Loss {}".format(running_loss / log_frequency))
			log_num += log_frequency
			running_loss = 0
		l = agent.train(dataset, minibatch_size)
		running_loss += l
		update += 1
	print("Training completed.")
	agent.checkpoint_network(env_name, extra_info)
	#Plot losses
	losses = []
	for loss in agent.losses:
		losses.append(loss.data.cpu().numpy())
	plot(losses, checkpoint_dir, env_name)
	#Evaluation
	print("beginning evaluation")
	evaluator = Evaluator(env_name, num_eval_episodes, checkpoint_dir, epsilon_greedy)
	evaluator.evaluate(agent)
	return agent

def train_transitions(env_name,
		minimal_action_set,
		learning_rate,
		alpha,
		l2_penalty,
		minibatch_size,
		hist_len,
		discount,
		checkpoint_dir,
		updates,
		dataset,
		num_eval_episodes):

	# create DQN agent
	agent = Imitator(list(minimal_action_set),
				learning_rate,
				alpha,
				checkpoint_dir,
				hist_len,
				l2_penalty)

	print("Beginning training...")
	log_frequency = 1000
	log_num = log_frequency
	update = 1
	running_loss = 0.
	while update < updates:
		if update > log_num:
			print(str(update) + " updates completed. Loss {}".format(running_loss / log_frequency))
			log_num += log_frequency
			running_loss = 0
		l = agent.train(dataset, minibatch_size)
		running_loss += l
		update += 1
	print("Training completed.")
	agent.checkpoint_network(env_name + "_transitions")
	#Plot losses
	losses = []
	for loss in agent.losses:
		losses.append(loss.data.cpu().numpy())
	plot(losses, checkpoint_dir, env_name + "_transitions")

	#calculate accuacy

	#Evaluation
	#evaluator = Evaluator(env_name, num_eval_episodes)
	#evaluator.evaluate(agent)
	return agent

if __name__ == '__main__':
	train()
