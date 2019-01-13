from ale_wrapper import ALEInterfaceWrapper
from preprocess import Preprocessor
from state import *
import numpy as np

class Evaluator:

	def __init__(self, rom, cap_eval_episodes=True, time_limit=60 * 60 * 30 / 4,
				action_repeat=4, hist_len=4, ale_seed=100, action_repeat_prob=0,
				num_eval_episodes=100):
		self.cap_eval_episodes = cap_eval_episodes
		self.time_limit = time_limit
		self.action_repeat = action_repeat
		self.hist_len = hist_len
		self.rom = rom
		self.ale_seed = ale_seed
		self.action_repeat_prob = action_repeat_prob
		self.num_eval_episodes = num_eval_episodes

	def evaluate(self, agent):
		ale = self.setup_eval_env(self.ale_seed, self.action_repeat_prob, self.rom)
		self.eval(ale, agent)

	def setup_eval_env(self, ale_seed, action_repeat_prob, rom):
		ale = ALEInterfaceWrapper(action_repeat_prob)
		#Set the random seed for the ALE

		ale.setInt('random_seed', ale_seed)
		'''
		This sets the probability from the default 0.25 to 0.
		It ensures deterministic actions.
		'''
		ale.setFloat('repeat_action_probability', action_repeat_prob)
		# Load the ROM file
		ale.loadROM(rom)
		return ale

	def eval(self, ale, agent):
		action_set = ale.getMinimalActionSet()
		rewards = []
		# 100 episodes
		for i in range(100):
			ale.reset_game()
			preprocessor = Preprocessor()
			state = State(self.hist_len)
			steps = 0
			utils.perform_no_ops(ale, 30, preprocessor, state)
			episode_reward = 0
			while not ale.game_over() and steps < self.time_limit:
				if np.random.uniform() < 0.05:
					action = np.random.choice(action_set)
				else:
					action = agent.get_action(state)
				for _ in range(self.action_repeat):
					episode_reward += ale.act(action)
					preprocessor.add(ale.getScreenRGB())
				state.add_frame(preprocessor.preprocess())
				steps += 1
			print "Episode " + str(i) + " reward is " + str(episode_reward)
			rewards.append(episode_reward)
		print "Mean reward is: " + str(np.mean(rewards))