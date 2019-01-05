import os
from bc import Imitator
from preprocess import Preprocessor
import numpy as np
from dataset import Example, Dataset
import utils

def train(training_frames,
		learning_rate,
		alpha,
		min_squared_gradient,
		minibatch_size,
		replay_capacity, 
		hist_len,
		discount,
		upd_freq, 
		replay_start_size, 
		checkpoint_dir,
		updates,
		dataset,
		l2_penalty):



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

	timestep = 0
	# Main training loop
	while timestep < training_frames:
		# create a state variable of size hist_len
		preprocessor = Preprocessor()
		# episode loop
		while not episode_done:
			if timestep % checkpoint_frequency == 0:
				epoch = timestep/checkpoint_frequency
				agent.checkpoint_network(epoch)

			action = agent.get_action(state.get_state())

			reward = 0
			#skip frames by repeating action
			for i in range(act_rpt):
				reward = reward + ale.act(action)
				#add the images on stack 
				preprocessor.add(ale.getScreenRGB())

			#increment episode reward before clipping the reward for training
			total_reward += reward
			reward = np.clip(reward, -1, 1)

			# get the preprocessed new frame
			img = preprocessor.preprocess()
			state.add_frame(img)

			#store transition
			replay_memory.add_item(img, action, reward, episode_done, time_since_term)

			'''
			Training. We only train once buffer has filled to 
			size=replay_start_size
			'''
			if (timestep > replay_start_size):
				# anneal epsilon.
				if timestep % eval_freq == 0:
					evaluator.evaluate(agent, timestep/eval_freq)
					ale.reset_game()
					# Break loop and start new episode after eval
					# Can help prevent getting stuck in episodes
					episode_done = True
				if timestep % upd_freq == 0:
					agent.train(replay_memory, minibatch_size) 

		episode_num = episode_num + 1

	if timestep == training_frames:
		evaluator.evaluate(agent, training_frames/eval_freq)
		agent.checkpoint_network(training/checkpoint_frequency)

if __name__ == '__main__':
	train()