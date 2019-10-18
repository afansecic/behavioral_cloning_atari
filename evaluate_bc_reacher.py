import argparse
import sys
import gym
from lowdim_bc_test import BCNet
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.insert(0,'./baselines/')
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.cmd_util import make_vec_env

from pylab import *


def bc_action(state, bc_net, device):
    with torch.no_grad():
        action = bc_net.forward(torch.from_numpy(state).float().to(device))
        #print(action)
    return action.cpu().numpy()


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_id', default='', help='Select the environment to run')
parser.add_argument('--env_type', default='', help='mujoco or atari')
parser.add_argument('--model_path', default='')
parser.add_argument('--bc_model_path', default='', help="path to pretrained bc policy")
parser.add_argument('--num_episodes', type=int, default=10, help="num epsiodes to run")
args = parser.parse_args()


env = make_vec_env(args.env_id, args.env_type, 1, 0,
                   wrapper_kwargs={
                       'clip_rewards':False,
                       'episode_life':False,
                   })
viewer = env.envs[0].unwrapped._get_viewer('human')

viewer.cam.trackbodyid = 0         # id of the body to track ()
viewer.cam.distance = env.envs[0].model.stat.extent * 1.0         # how much you "zoom in", model.stat.extent is the max limits of the arena
viewer.cam.lookat[0] += 0.0         # x,y,z offset from the object (works if trackbodyid=-1)
viewer.cam.lookat[1] += 0.0
viewer.cam.lookat[2] += 0.0
viewer.cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
viewer.cam.azimuth = 0              # camera rotation around the camera's vertical axis

env = VecNormalize(env,ob=True,ret=False,eval=True)
#
try:
    env.load(args.model_path) # Reload running mean & rewards if available
    print("loaded env")
except AttributeError:
    print("failed to load vec normalize")
    input("Continue?")
    pass
#env.render()



#load BCNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bc_policy = BCNet(11,2)
bc_policy.load_state_dict(torch.load(args.bc_model_path))
bc_policy.to(device)

for episode in range(args.num_episodes):
    env.venv.unwrapped.envs[0].unwrapped.seed(episode)
    state, done = env.reset(), False
    steps = 0
    acc_reward = 0
    while not done:
        a = bc_action(state, bc_policy, device)
        #print(a)
        state, r, done, _ = env.step(a)
        steps += 1
        acc_reward += r
        env.render()
        pause(1/30)
    print(steps,acc_reward)
        #pixel = env.render('rgb_array')
        #imageio.imwrite('pic' + str(cnt) +'.jpg', pixel)
