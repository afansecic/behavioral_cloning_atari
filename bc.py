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
				hist_len):
		self.minimal_action_set = min_action_set
		self.chkpt_freq = checkpoint_frequency
		self.network = Network(len(self.minimal_action_set))
		if torch.cuda.is_available():
			print "Initializing Cuda Nets..."
			self.prediction_net.cuda()
			self.target_net.cuda()
		self.optimizer = optim.RMSprop(self.prediction_net.parameters(),
		lr=learning_rate, alpha=alpha, eps=min_squared_gradient)
		self.checkpoint_directory = checkpoint_dir


	def predict(self, state):
			# predict action probabilities
			outputs = self.network(Variable(utils.float_tensor(state)))
			probabilities = outputs[len(outputs) - 1].data.cpu().numpy()
			return probabilities

	def get_action(self, state):
		probabilities = self.predict(state)
		action = np.random.choice(self.minimal_action_set, p=probabilities)
		return action

	# potentially optimizable
	def compute_labels(self, sample, minibatch_size):
		label = Variable(utils.float_tensor(minibatch_size))
		for i in range(minibatch_size):
			if (sample[i].game_over):
				label[i] = sample[i].reward
			else:
				state = Variable(utils.float_tensor(sample[i].new_state))
				outputs = self.target_net(state)
				target_q_vals = outputs[len(outputs)-1]
				label[i] = sample[i].reward + self.discount * torch.max(target_q_vals).data[0]
		return label

	'''
	Args - 
	outputs: a batchsize x #actions matrix of Q-values,
	where each row contains the Q-values for the actions
	action_indices: contains batchsize indexes. action_indices[i] index 
	corresponds to the index of action taken for the ith state in the 
	minibatch
	targets: contains the target Q-values. 
	'''
	def get_loss(self, outputs, action_indices, targets):
		'''
		creates matrix of shape #actions x batchsize filled 
		with 0s.
		'''
		one_hot_mat = Variable(utils.float_tensor(outputs.size()[::-1]).zero_())
		'''
		Each column i represents one state from the minibatch.
		Place a 1.0 at at the index of the action taken in that 
		state.
		'''
		for i in range(len(action_indices)):
			one_hot_mat[action_indices[i], i] = 1.0
		'''
		The ith row of outputs contains the vector Q(s_i, A),
		the Q-values for all actions for the ith state in the 
		minibatch.
		Thus, multiplying the ith row of 'outputs' by the ith
		column of one_hot_mat, the result is Q(s_i, a_i),
		the predicted Q-value for the ith state-action pair
		in the minibatch. Multiplying outputs and one_hot_mat
		gives the Q-values of the minibatch on the diagonal.
		'''
		q_vals = torch.diag(torch.mm(outputs, one_hot_mat))
		return nn.SmoothL1Loss()(q_vals, targets)

	'''
	Args:
	replay_memory: An instance of the ReplayMemory class
	defined in replaybuffer.py
	minibatch_size: The size of the minibatch
	save_tuple: A tuple containing two values - a boolean
	indicating whether to checkpoint the network, and the
	epoch number.
	'''
	def train(self, replay_memory, minibatch_size):
		# sample a minibatch of transitions
		sample = replay_memory.sample_minibatch(minibatch_size, self.rnd_buffer_sample)
		# compute the target values for the minibatch
		labels = self.compute_labels(sample, minibatch_size)
		state = Variable(utils.float_tensor(np.stack([np.squeeze(x.state) for x in sample])))
		# The list of ALE actions taken for the minibatch
		actions_taken = [x.action for x in sample]
		# The indices of the ALE actions taken in the action set
		action_indices = [self.minimal_action_set.index(x) for x in actions_taken]
		
		self.optimizer.zero_grad()
		'''
		Forward pass the minibatch through the 
		prediction network.
		'''
		activations = self.prediction_net(state)
		'''
		Extract the Q-value vectors of the minibatch
		from the final layer's activations. See return values
		of the forward() functions in cnn.py
		'''
		output = activations[len(activations) - 1]
		loss = self.get_loss(output, action_indices, labels)
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