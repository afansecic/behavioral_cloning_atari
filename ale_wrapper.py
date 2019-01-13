from ale_python_interface import ALEInterface
from copy import deepcopy
import numpy as np

class ALEInterfaceWrapper:
	def __init__(self, repeat_action_probability):
		self.internal_action_repeat_prob = repeat_action_probability
		self.prev_action = 0
		self.ale = ALEInterface()
		'''
		This sets the probability from the default 0.25 to 0.
		It ensures deterministic actions.
		'''
		self.ale.setFloat('repeat_action_probability', repeat_action_probability)

	def getScreenRGB(self):
		return self.ale.getScreenRGB()
		
	def game_over(self):
		return self.ale.game_over()

	def reset_game(self):
		self.ale.reset_game()

	def lives(self):
		return self.ale.lives()

	def getMinimalActionSet(self):
		return self.ale.getMinimalActionSet()

	def setInt(self, key, value):
		self.ale.setInt(key, value)

	def setFloat(self, key, value):
		self.ale.setFloat(key, value)

	def loadROM(self, rom):
		self.ale.loadROM(rom)

	def act(self, action):
		actual_action = action
		return self.ale.act(actual_action)
