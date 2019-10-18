import argparse
import sys

import gym
from gym import wrappers, logger
import tensorflow as tf
from bc import Clone
import numpy as np
import torch

###Code to run BC evaluations and generate videos of BC behavior



sys.path.append('./baselines/')
from baselines.common.trex_utils import preprocess
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment to run, e.g. breakout')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--episode_count', default=100, type=int)
    parser.add_argument('--record_video', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default = 123, type=int)
    parser.add_argument('--epsilon_greedy', default=0.01, help="Probability of taking random action", type=float)
    args = parser.parse_args()
    print(args)

    env_name = args.env_name
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    #env = gym.make(args.env_id)

    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, args.env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })
    env.envs[0].unwrapped.seed(seed)
    env.action_space.np_random.seed(seed)

    if args.record_video:
        env = VecVideoRecorder(env,'./videos/',lambda steps: True, 20000) # Always record every episode

    if args.env_type == 'atari':
        env = VecFrameStack(env, 4)
    elif args.env_type == 'mujoco':
        env = VecNormalize(env,ob=True,ret=False,eval=True)
    else:
        assert False, 'not supported env type'

    #get all available actions
    min_action_set =set([i for i in range(env.action_space.n)])

    #create BC agent using pretrained policy
    agent = Clone(min_action_set, 4, args.model_path)


    for i in range(args.episode_count):
        ob = env.reset()
        steps = 0
        acc_reward = 0
        while True:

            #preprocess the state
            state = preprocess(ob, env_name)
            state = np.transpose(state, (0, 3, 1, 2))
            r = np.random.rand()
            #print(r)
            if r < args.epsilon_greedy:
                #print('eps greedy action',r)
                action = env.action_space.sample()
            else:
                #print('policy action')
                action = agent.get_action(state)
            #print("action", action)
            ob, reward, done, _ = env.step(action)
            if args.render:
                env.render()

            steps += 1
            acc_reward += reward
            if done:
                print(steps,acc_reward)
                break
        #print(steps,acc_reward)
    env.close()
    env.venv.close()
