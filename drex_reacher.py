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
import run_test

from pylab import *

#learn reward with noise injection


def bc_action(state, bc_net, device):
    with torch.no_grad():
        action = bc_net.forward(torch.from_numpy(state).float().to(device))
        #print(action)
    return action.cpu().numpy()

class Net(nn.Module):
    def __init__(self, obs_in):
        super().__init__()

        self.fc1 = nn.Linear(obs_in, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,32)
        #self.fc1 = nn.Linear(1936,64)
        self.fc4 = nn.Linear(32, 1)



    def forward(self, traj):
        #assumes traj is of size [batch, height, width, channel]
        #this formatting should be done before feeding through network
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #print(traj.shape)
        #compute forward pass of reward network
        x = torch.relu(self.fc1(traj))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        #print(x)
        #r = torch.tanh(self.fc2(x)) #clip reward?
        #r = F.celu(self.fc2(x))
        rs = -torch.relu(self.fc4(x))
        #print(rs)
        r = torch.sum(rs)
        return r



#Takes as input a list of lists of demonstrations where first list is lowest ranked and last list is highest ranked
def create_training_data_from_bins(ranked_demos):


    #n_train = 3000 #number of pairs of trajectories to create
    #snippet_length = 50
    training_obs = []
    training_labels = []
    num_ranked_bins = len(ranked_demos)
    #pick progress based snippets
    for i in range(num_ranked_bins):
        for ti in ranked_demos[i]:
            for j in range(i+1, num_ranked_bins):
                for tj in ranked_demos[j]:
                    training_obs.append((ti, tj))
                    training_labels.append(1)


    return training_obs, training_labels


def learn_reward(reward_network, training_inputs, training_outputs, optimizer, num_iter):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #check if gpu available
    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    training_dataset = list(zip(training_inputs, training_outputs))
    #partition into training and validation sets with 90/10 split

    for epoch in range(num_iter):
        print("epoch", epoch)
        np.random.shuffle(training_dataset)
        training_obs, training_labels = zip(*training_dataset)
        cum_loss = 0.0
        num_correct = 0
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)


            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            logits_i = reward_network.forward(traj_i)
            logits_j = reward_network.forward(traj_j)
            cum_returns = torch.cat([logits_i.unsqueeze(0), logits_j.unsqueeze(0)],0)
            loss = loss_criterion(cum_returns.unsqueeze(0), labels)

            loss.backward()
            optimizer.step()

            pred_label = np.argmax([logits_i.item(), logits_j.item()])
            #print(outputs)
            #_, pred_label = torch.max(outputs,0)
            #print(pred_label)
            #print(label)
            if pred_label == labels.item():
                num_correct += 1.

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 5000 == 0:
                print(i)
                print("cum loss", cum_loss)
                print("accuracy in batch", num_correct/5000)
                num_correct = 0
                cum_loss = 0.0
                torch.save(reward_net.state_dict(), "./learned_models/reacher_trex_reward.params")
        #accuracy = calc_accuracy(reward_network, training_inputs, training_outputs)
        #print("accuracy", accuracy)

    print("finished training")





def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            traj_i, traj_j = training_inputs[i]
            label = training_outputs[i]
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)



            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            logits_i = reward_network.forward(traj_i)
            logits_j = reward_network.forward(traj_j)
            #print(outputs_i, outputs_j)
            #print("label", label)
            pred_label = np.argmax([logits_i.item(), logits_j.item()])
            #print(outputs)
            #_, pred_label = torch.max(outputs,0)
            #print(pred_label)
            #print(label)
            if pred_label == label:
                num_correct += 1.
    return num_correct / len(training_inputs)






def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.forward(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))



parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_id', default='', help='Select the environment to run')
parser.add_argument('--env_type', default='', help='mujoco or atari')
parser.add_argument('--model_path', default='')
parser.add_argument('--bc_model_path', default='', help="path to pretrained bc policy")
parser.add_argument('--num_episodes', type=int, default=10, help="num epsiodes to run")
parser.add_argument('--num_demos', type=int, default=3, help="num demos to give")
parser.add_argument('--render', action='store_true')
parser.add_argument('--seed', default=0, type=int, help="random seed for experiments")
args = parser.parse_args()

lr = 0.00001
weight_decay = 0.00001
num_iter = 15

seed = int(args.seed)
torch.manual_seed(seed)
np.random.seed(seed)

env = make_vec_env(args.env_id, args.env_type, 1, 0,
                   wrapper_kwargs={
                       'clip_rewards':False,
                       'episode_life':False,
                   })
if args.render:
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

#load the demonstrations (demo agent)
agent = run_test.PPO2Agent(env,args.env_type,False)
agent.load(args.model_path)

#Generate the demonstrations as the best category
#record the data for behavioral cloning
demonstrations = []
for i in range(args.num_demos):
    env.venv.unwrapped.envs[0].unwrapped.seed(i)
    ob = env.reset()
    trajectory = []
    steps = 0
    acc_reward = 0
    while True:
        trajectory.append(ob[0])
        #get grayscale image
        action = agent.act(ob)

        ob, reward, done, _ = env.step(action)

        steps += 1
        acc_reward += reward

        if done:
            #print(steps,acc_reward)
            break

    demonstrations.append(np.array(trajectory))

#load BCNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bc_policy = BCNet(11,2)
bc_policy.load_state_dict(torch.load(args.bc_model_path))
bc_policy.to(device)

epsilons = [1.0, 0.75, 0.5, 0.25, 0.05]
ranked_bins = []
for epsilon in epsilons:
    #print("epsilon", epsilon)
    noise_bin = []
    for episode in range(args.num_episodes):
        #print("episode", episode)

        env.venv.unwrapped.envs[0].unwrapped.seed(episode)
        state, done = env.reset(), False
        steps = 0
        acc_reward = 0
        trajectory = []
        while not done:
            trajectory.append(state[0])
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = bc_action(state, bc_policy, device)
            #print(a)
            state, r, done, _ = env.step(a)
            steps += 1
            acc_reward += r
            if args.render:
                env.render()
                pause(1/30)
        #print(steps,acc_reward)

        noise_bin.append(np.array(trajectory))
    ranked_bins.append(noise_bin)

#add demos as best
ranked_bins.append(demonstrations)

#run T-REX
#create all pairs of trajectories



print("Learning from ", len(ranked_bins), "synthetically ranked batches of demos")

# Now we create a reward network and optimize it using the training data.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = Net(11)
reward_net.to(device)
import torch.optim as optim
training_obs, training_labels = create_training_data_from_bins(ranked_bins)
print("training on ", len(training_obs), "auto labeled pairs")
#print("training_obs", training_obs)
#print("training_labels",training_labels)
#learn reward net
optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
learn_reward(reward_net, training_obs, training_labels, optimizer, num_iter)
torch.save(reward_net.state_dict(), "./learned_models/reacher_trex_reward.params")

#print out the predicted returns for the noise injection
with torch.no_grad():
    for i in range(len(ranked_bins)):
        print("bin", i)
        for t in ranked_bins[i]:
            traj_return = reward_net.forward(torch.from_numpy(t).float().to(device))
            print(traj_return.item())
