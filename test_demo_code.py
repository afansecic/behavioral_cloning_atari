import numpy as np
import os
storage_dir = "/home/dsbrown/Code/behavioral_cloning_atari/demos/BreakoutNoFrameskip-v4"
save_number = 0
actions = np.load(os.path.join(storage_dir, "actions" + str(save_number) + ".npy"))
states = np.load(os.path.join(storage_dir, "states" + str(save_number) + ".npy"))
print(actions[-500:])
print(states.shape)
