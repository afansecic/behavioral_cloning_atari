import argparse
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
print(gym.__file__)

from gym import wrappers, logger
import matplotlib.pyplot as plt

sys.path.insert(0,'./baselines/')
from baselines.ppo2.model import Model
from baselines.common.policies import build_policy
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from skimage import color


from pylab import *

#I want to see if we can do behavioral cloning from three demonstrations in reacher

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class PPO2Agent(object):
    def __init__(self, env, env_type, stochastic=False):
        ob_space = env.observation_space
        ac_space = env.action_space

        if env_type == 'atari':
            policy = build_policy(env,'cnn')
        elif env_type == 'mujoco':
            policy = build_policy(env,'mlp')

        make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=1,
                        nsteps=1, ent_coef=0., vf_coef=0.,
                        max_grad_norm=0.)
        self.model = make_model()
        self.stochastic = stochastic

    def load(self, path):
        self.model.load(path)

    def act(self, observation):
        if self.stochastic:
            a,v,state,neglogp = self.model.step(observation)
        else:
            a = self.model.act_model.act(observation)
        return a

def calculate_validation_loss(bc_net, validation_sa, loss_criterion):
    validation_obs = np.array([s for s,_ in validation_sa])
    validation_labels = np.array([a for _,a in validation_sa])
    with torch.no_grad():
        batch_obs = torch.from_numpy(validation_obs).float().to(device)
        batch_labels = torch.from_numpy(validation_labels).float().to(device)

        pred_actions = bc_net.forward(batch_obs)
        loss = loss_criterion(pred_actions, batch_labels)
    return loss.item()


class BCNet(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 32)
        #self.fc1 = nn.Linear(1936,64)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)



    def forward(self, x):
        #run behavioral cloning through network
        sum_rewards = 0
        sum_abs_rewards = 0
        #print(traj.shape)
        #compute forward pass of reward network
        out1 = F.leaky_relu(self.fc1(x))
        out2 = F.leaky_relu(self.fc2(out1))
        actions = self.fc3(out2)
        return actions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--episode_count', type=int, default=100)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--bc_model_save_path', type=str, default="./learned_models/reacher_bc_net.params")
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    #env = gym.make(args.env_id)

    #env id, env type, num envs, and seed
    env = make_vec_env(args.env_id, args.env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })

    from mujoco_py import GlfwContext
    GlfwContext(offscreen=True)
    viewer = env.envs[0].unwrapped._get_viewer('rgb_array')

    viewer.cam.trackbodyid = 0         # id of the body to track ()
    viewer.cam.distance = env.envs[0].model.stat.extent * 0.49        # how much you "zoom in", model.stat.extent is the max limits of the arena
    viewer.cam.lookat[0] += 0.0         # x,y,z offset from the object (works if trackbodyid=-1)
    viewer.cam.lookat[1] += 0.0
    viewer.cam.lookat[2] += 0.0
    viewer.cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
    viewer.cam.azimuth = 0              # camera rotation around the camera's vertical axis


    #Need this for RL policy to work
    env = VecNormalize(env,ob=True,ret=False,eval=True)
    #
    try:
        env.load(args.model_path) # Reload running mean & rewards if available
    except AttributeError:
        print("failed to load vec normalize running mean and std")
        input("Continue?")
        pass

    agent = PPO2Agent(env,args.env_type,args.stochastic)
    agent.load(args.model_path)
    #agent = RandomAgent(env.action_space)

    episode_count = args.episode_count
    reward = 0
    done = False

    #record the data for behavioral cloning
    demonstrated_sa = []
    for i in range(episode_count):
        env.venv.unwrapped.envs[0].unwrapped.seed(i)
        ob = env.reset()
        trajectory_sa = []
        steps = 0
        acc_reward = 0
        while True:
            if args.render:
                env.render()
                pause(1/30)
            #get grayscale image
            action = agent.act(ob)

            demonstrated_sa.append((ob[0], action[0]))

            ob, reward, done, _ = env.step(action)

            steps += 1
            acc_reward += reward

            if done:
                print(steps,acc_reward)
                break

        # plt.imshow(rgb_img)
        # plt.show()

    #env.close()
    #env.venv.close()

    #take the demonstrations and learn classification
    print(demonstrated_sa[0])
    print(demonstrated_sa[0][0])
    print(demonstrated_sa[0][1])
    observation_size = len(demonstrated_sa[0][0])
    action_size = len(demonstrated_sa[0][1])
    print(observation_size, action_size)

    #create data for training network
    training_obs = np.array([s for s,_ in demonstrated_sa])
    print(training_obs.shape)
    training_labels = np.array([a for _,a in demonstrated_sa])
    print(training_labels.shape)
    batch_size = args.batch_size
    bc_net = BCNet(observation_size, action_size)


    #debug observations
    # print(training_obs)
    # for i in range(observation_size):
    #     plt.plot(training_obs[:,i], label='feature ' + str(i))
    # plt.legend()
    # plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    bc_net.to(device)

    validation_percent = 0.0
    validation_sa = demonstrated_sa[:int(np.round(validation_percent * len(demonstrated_sa)))]
    training_sa = demonstrated_sa[int(np.round(validation_percent * len(demonstrated_sa))) :]
    print("splitting into train (95/%) and val sets (5%)")
    print(len(validation_sa))
    print(len(training_sa))

    optimizer = optim.Adam(bc_net.parameters(),  lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_criterion = torch.nn.MSELoss()
    count = 0
    best_v_loss = np.float('inf')
    for iter in range(args.iterations):
        np.random.shuffle(training_sa)
        training_obs = np.array([s for s,_ in training_sa])
        training_labels = np.array([a for _,a in training_sa])
        print("iter", iter)
        cum_loss = 0.0
        for i in range(len(training_sa) - batch_size):
            #create batch and train net
            batch_obs = torch.from_numpy(training_obs[i:i+batch_size, :]).float().to(device)
            batch_labels = torch.from_numpy(training_labels[i:i+batch_size, :]).float().to(device)
            optimizer.zero_grad()
            pred_actions = bc_net.forward(batch_obs)
            loss = loss_criterion(pred_actions, batch_labels)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        print("train_loss", cum_loss)
        if len(validation_sa) > 0:
            val_loss = calculate_validation_loss(bc_net, validation_sa, loss_criterion)
            print("validation loss", val_loss)
            if val_loss < best_v_loss:
                print("check pointing")
                torch.save(bc_net.state_dict(), args.bc_model_save_path)
                count = 0
                best_v_loss = val_loss
            else:
                count += 1
                if count > 20:
                    print("Stopping to prevent overfitting after {} ".format(count))
                    #early_stop = True
                    break
        else:
            #save model every episode
            torch.save(bc_net.state_dict(), args.bc_model_save_path)
    #test out predictions
    for i in range(4):
        print("-"*10)
        print(i)
        obs = training_obs[i]
        print(obs)
        action = training_labels[i]
        print(action)
        with torch.no_grad():
            pred = bc_net.forward(torch.from_numpy(obs).float().to(device).unsqueeze(0))
        print(pred)
