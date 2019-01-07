import torch
import argparse
import numpy as np
from train import train
from pdb import set_trace
import episodic_segmentation

def print_args(args, file):
	arguments = vars(args)
	for arg in arguments:
		file.write(str(arg) + ":" + str(arguments[arg]) + "\n")
	file.flush()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--rom",
		default="/u/prabhatn/ale/thesis/dqn/roms/pong.bin",
		type=str, required=True)

	'''
	Random seed for the Arcade Learning Environment
	'''
	parser.add_argument("--ale-seed", type=int, default=123)

	# ##################################################
	# ##             Algorithm parameters             ##
	# ##################################################
	parser.add_argument("--updates", type=int, default=75000)
	parser.add_argument("--minibatch-size", type=int, default=32)
	parser.add_argument("--hist-len", type=int, default=4)
	parser.add_argument("--discount", type=float, default=0.99)
	parser.add_argument("--learning-rate", type=float, default=0.00025)

	parser.add_argument("--alpha", type=float, default=0.95)
	parser.add_argument("--min-squared-gradient", type=float, default=0.01)

	# ##################################################
	# ##                   Files                      ##
	# ##################################################

	parser.add_argument('--storage-dir',
		default="/scratch/cluster/prabhatn/imitation/offline/pong0/storage", 
		required=True)
	parser.add_argument('--storage-size',
		default=200000,
		type=int, required=True)

	args = parser.parse_args()

	args_file = open(args.args_output_file, "w")
	print_args(args, args_file)


	storage_dir = args.storage_dir # goscratch and imitation dir
	batch_storage_size = args.storage_size #200000
	episodes = episodic_segmentation.extract_episodes(storage_dir, batch_storage_size)
	episodes = sorted(episodes, key=lambda x: (x[2], x[0]))
	set_trace()
	# episodes has tuples of the form start time, end time, total reward
	# start, end, total_reward = episodes[0]
	# episode = Episode(episodes[0][0], episodes[0][1], batch_storage_size, storage_dir, 4)
	# # episode[i] returns a dict (state, action, reward, terminal, next_state)
	# # get first step of episode	
	# transition = episode[0]
	# print transition['state']
	# print transition['next_state']
	# print transition['reward']
	# print transition['action']
	# print transition['terminal']
	# print sum([episode[i]['terminal'] for i in range(episode.episode_len)])

	train(args.rom,
		args.ale_seed,
		args.learning_rate,
		args.alpha,
		args.min_squared_gradient,
		args.l2_penalty,
		args.minibatch_size,
		args.hist_len,
		args.discount,
		args.checkpoint_frequency,
		args.updates,
		dataset)