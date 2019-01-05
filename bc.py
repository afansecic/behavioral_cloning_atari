#!/usr/bin/env python
from cnn import Network
import numpy as np
import utils
import copy
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable

class Imitator:
	def __init__(self, min_action_set,
				learning_rate,
				alpha,
				min_squared_gradient,
				checkpoint_frequency,
				checkpoint_dir,
				hist_len,
				l2_penalty):
		self.minimal_action_set = min_action_set
		self.chkpt_freq = checkpoint_frequency
		self.network = Network(len(self.minimal_action_set))
		if torch.cuda.is_available():
			print "Initializing Cuda Nets..."
			self.prediction_net.cuda()
			self.target_net.cuda()
		self.optimizer = optim.RMSprop(self.prediction_net.parameters(),
		lr=learning_rate, alpha=alpha, eps=min_squared_gradient, weight_decay=l2_penalty)
		self.checkpoint_directory = checkpoint_dir


	def predict(self, state):
		# predict action probabilities
		outputs = self.network(Variable(utils.float_tensor(state)))
		vals = outputs[len(outputs) - 1].data.cpu().numpy()
		return vals

	def get_action(self, state):
		vals = self.predict(state)
		return self.minimal_action_set[np.argmax(vals)]

	# potentially optimizable
	def compute_labels(self, sample, minibatch_size):
		labels = Variable(utils.int_tensor(minibatch_size))
		# The list of ALE actions taken for the minibatch
		actions_taken = [x.action for x in sample]
		# The indices of the ALE actions taken in the action set
		action_indices = [self.minimal_action_set.index(x) for x in actions_taken]
		for index in range(len(action_indices)):
			labels[index] = action_indices[index]
		return labels

	'''
	Args - 
	outputs: a batchsize x #actions matrix of Q-values,
	where each row contains the Q-values for the actions
	action_indices: contains batchsize number of indexes. action_indices[i]
	corresponds to the index of action taken for the ith state in the 
	minibatch
	'''
	def get_loss(self, outputs, labels):
		return nn.CrossEntropyLoss()(outputs, labels)

	'''
	Args:
	replay_memory: An instance of the ReplayMemory class
	defined in replaybuffer.py
	minibatch_size: The size of the minibatch
	save_tuple: A tuple containing two values - a boolean
	indicating whether to checkpoint the network, and the
	epoch number.
	'''
	def train(self, dataset, minibatch_size):
		# sample a minibatch of transitions
		sample = dataset.sample_minibatch(minibatch_size)
		state = Variable(utils.float_tensor(np.stack([np.squeeze(x.state) for x in sample])))
		
		# compute the target values for the minibatch
		labels = self.compute_labels(sample, minibatch_size)

		self.optimizer.zero_grad()
		'''
		Forward pass the minibatch through the 
		prediction network.
		'''
		activations = self.network(state)
		'''
		Extract the Q-value vectors of the minibatch
		from the final layer's activations. See return values
		of the forward() functions in cnn.py
		'''
		output = activations[len(activations) - 1]
		loss = self.get_loss(output, labels)
		loss.backward()
		self.optimizer.step()

	'''
	Args:
	epoch - the training epoch number
	This function checkpoints the prediction
	network.
	'''
	def checkpoint_network(self, epoch):
		print "Checkpointing Weights"
		utils.save_checkpoint({
			'epoch': epoch, 
			'state_dict': self.prediction_net.state_dict()
			}, epoch, self.checkpoint_directory)
		print "Checkpointed."