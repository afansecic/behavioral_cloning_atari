#from ale_wrapper import ALEInterfaceWrapper
from preprocess import Preprocessor
from state import *
import numpy as np
import utils
import gym
from baselines.ppo2.model import Model
from baselines.common.policies import build_policy
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
import torch
import argparse
import numpy as np
from train import train
from pdb import set_trace
import dataset
import tensorflow as tf
from bc import Clone
from baselines.common.trex_utils import preprocess



class DemoGenerator:

    def __init__(self, agent, env_name, num_eval_episodes, seed):
        self.agent = agent
        self.env_name = env_name
        self.num_eval_episodes = num_eval_episodes
        self.seed = seed

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
        env_type = "atari"
        #env id, env type, num envs, and seed
        env = make_vec_env(env_id, env_type, 1, seed, wrapper_kwargs={'clip_rewards':False,'episode_life':False,})
        if env_type == 'atari':
            env = VecFrameStack(env, 4)

        print("env actions", env.action_space)
        self.env = env

    def get_pseudo_rankings(self, epsilon_greedy_list):
        ranked_batches = []
        for epsilon_greedy in epsilon_greedy_list:
            demo_batch = self.generate_demos(self.env, self.agent, epsilon_greedy)
            ranked_batches.append(demo_batch)
        return ranked_batches


    def generate_demos(self, env, agent, epsilon_greedy):
        print("Generating demos for epsilon=",epsilon_greedy)
        rewards = []
        # 100 episodes
        episode_count = self.num_eval_episodes
        reward = 0
        done = False
        rewards = []
        cum_steps = []
        demos = []
        #writer = open(self.checkpoint_dir + "/" +self.env_name + "_bc_results.txt", 'w')
        for i in range(int(episode_count)):
            ob = env.reset()
            steps = 0
            acc_reward = 0
            traj = []
            while True:
                #preprocess the state
                state = preprocess(ob, self.env_name)
                traj.append(state)
                state = np.transpose(state, (0, 3, 1, 2))
                if np.random.rand() < epsilon_greedy:
                    #print('eps greedy action')
                    action = env.action_space.sample()
                else:
                    #print('policy action')
                    action = agent.get_action(state)
                ob, reward, done, _ = env.step(action)
                steps += 1
                acc_reward += reward
                if done:
                    print("Episode: {}, Steps: {}, Reward: {}".format(i,steps,acc_reward))
                    #writer.write("{}\n".format(acc_reward[0]))
                    rewards.append(acc_reward)
                    cum_steps.append(steps)
                    break
            demos.append(traj)

        print("Mean reward is: " + str(np.mean(rewards)))
        print("Mean step length is: " + str(np.mean(cum_steps)))
        return demos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ##################################################
    # ##             Algorithm parameters             ##
    # ##################################################
    #parser.add_argument("--dataset-size", type=int, default=75000)
    #parser.add_argument("--updates", type=int, default=10000)#200000)
    parser.add_argument("--env_name", type=str, help="Atari environment name in lowercase, i.e. 'beamrider'")
    parser.add_argument("--checkpoint_policy", type=str)
    parser.add_argument("--num_eval_episodes", type=int, default = 20)
    parser.add_argument('--seed', default=0, help="random seed for experiments")

    epsilon_greedy_list = [0.01, 0.1, 0.3, 0.5, 1.0]

    hist_length = 4
    args = parser.parse_args()

    seed = int(args.seed)
    print("seed", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

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

    env_type = "atari"
    #TODO: minimal action set from env
    minimal_action_set = [0,1,2,3]

    agent = Clone(list(minimal_action_set), hist_length, args.checkpoint_policy)
    print("beginning evaluation")
    generator = DemoGenerator(agent, env_name, args.num_eval_episodes, seed)
    ranked_demos = generator.get_pseudo_rankings(epsilon_greedy_list)
    print(len(ranked_demos))
