import torch
import argparse
import numpy as np
from train import train
from pdb import set_trace
import episode_segmentation
import dataset
from episode_segmentation import Episode

def print_args(args, file):
	arguments = vars(args)
	for arg in arguments:
		file.write(str(arg) + ":" + str(arguments[arg]) + "\n")
	file.flush()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--rom",
		default="/u/prabhatn/ale/thesis/dqn/roms/pong.bin",
		type=str)

	'''
	Random seed for the Arcade Learning Environment
	'''
	parser.add_argument("--ale-seed", type=int, default=123)
	parser.add_argument("--action-repeat-probability", type=float, default=0.0)

	# ##################################################
	# ##             Algorithm parameters             ##
	# ##################################################
	parser.add_argument("--dataset-size", type=int, default=75000)
	parser.add_argument("--updates", type=int, default=200000)
	parser.add_argument("--minibatch-size", type=int, default=32)
	parser.add_argument("--hist-len", type=int, default=4)
	parser.add_argument("--discount", type=float, default=0.99)
	parser.add_argument("--learning-rate", type=float, default=0.00025)

	parser.add_argument("--alpha", type=float, default=0.95)
	parser.add_argument("--min-squared-gradient", type=float, default=0.01)
	parser.add_argument("--l2-penalty", type=float, default=0.00001)
	parser.add_argument("--checkpoint-dir", type=str, default="/u/prabhatn/behavioral_cloning_atari/checkpoints")

	# ##################################################
	# ##                   Files                      ##
	# ##################################################

	parser.add_argument('--storage-dir',
		default="/scratch/cluster/prabhatn/imitation/offline/pong0/storage")
	parser.add_argument('--storage-size',
		default=200000,
		type=int)

	args = parser.parse_args()

	# args_file = open(args.args_output_file, "w")
	# print_args(args, args_file)


	storage_dir = args.storage_dir # goscratch and imitation dir
	batch_storage_size = args.storage_size #200000
	episodes = episode_segmentation.extract_episodes(storage_dir, batch_storage_size)
	episodes = sorted(episodes, key=lambda x: (x[2], x[0]))
	# episodes has tuples of the form start time, end time, total reward
	start, end, total_reward = episodes[0]
	episode = Episode(episodes[0][0], episodes[0][1], batch_storage_size, storage_dir, 4)
	# episode[i] returns a dict (state, action, reward, terminal, next_state)
	# get first step of episode	
	num_episodes = len(episodes)
	demo_episodes = [Episode(episodes[i][0],episodes[i][1],
							batch_storage_size,
							storage_dir, 4) 
							for i in range(num_episodes - 10,
										num_episodes-1)]

	data = dataset.Dataset(args.dataset_size, args.hist_len)
	episode_index_counter = 0
	dataset_size = 0
	for episode in demo_episodes:
		for index in range(episode.episode_len):
			transition = episode[index]
			state = transition['state']
			action = transition['action']
			data.add_item(state, action)
			dataset_size += 1
			if dataset_size == args.dataset_size:
				break
		if dataset_size == args.dataset_size:
			break

	train(args.rom,
		args.ale_seed,
		args.action_repeat_probability,
		args.learning_rate,
		args.alpha,
		args.min_squared_gradient,
		args.l2_penalty,
		args.minibatch_size,
		args.hist_len,
		args.discount,
		args.checkpoint_dir,
		args.updates,
		data)