import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20,20)
        #self.fc1 = nn.Linear(1936,64)
        self.fc3 = nn.Linear(20, 1)



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
        #print(x)
        #r = torch.tanh(self.fc2(x)) #clip reward?
        #r = F.celu(self.fc2(x))
        rs = self.fc3(x)
        #print(rs)
        r = torch.sum(rs)
        return r

def create_demo_bins(epsilons, bc_controller, demo_start, dt, T, num_rollouts = 20):
    demo_bins = []
    for n in range(len(epsilons)):
        bin = []
        for i in range(num_rollouts):
            xbcs, ubcs = get_bc_traj(bc_controller, demo_start, dt, T, epsilons[n])
            bin.append(xbcs)
        demo_bins.append(bin)
    return demo_bins


#Takes as input a list of lists of demonstrations where first list is lowest ranked and last list is highest ranked
def create_training_data_from_bins(ranked_demos):


    step = 4
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
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float()
            traj_j = torch.from_numpy(traj_j).float()
            labels = torch.from_numpy(labels)


            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            logits_i = reward_network.forward(traj_i)
            logits_j = reward_network.forward(traj_j)
            cum_returns = torch.cat([logits_i.unsqueeze(0), logits_j.unsqueeze(0)],0)
            loss = loss_criterion(cum_returns.unsqueeze(0), labels)

            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
        print("cum loss", cum_loss)

    print("finished training")



def get_demonstration(init_x, k, dt, T, noise_std, goal = np.array([0.0, 0.0])):
    xs = np.zeros((T, 2))
    us = np.zeros((T, 2))
    x = init_x
    for i in range(T):
        xs[i,:] = x
        noise = noise_std * np.random.randn(2)
        #print("noise", noise)
        #print("x", x)
        u = k * (goal - x) + noise
        #print("u", u)
        x = x + u * dt
        us[i,:] = u
    return xs, us

def get_bc_traj(bc_controller, init_x, dt, T, eps_greedy = 0.0, eps_k = 0.5):

    #rerun from initial position using BC policy
    xbcs = np.zeros((T, 2))
    ubcs = np.zeros((T, 2))
    x = init_x
    for i in range(T):
        xbcs[i,:] = x
        if np.random.rand() < eps_greedy:
            #take random action
            rand_theta = 2 * np.pi * np.random.rand()
            u = eps_k * np.array([np.cos(rand_theta), np.sin(rand_theta)])
        else:
            u = np.dot(np.append(x,[1.0]),bc_controller)
        #print("u", u)
        x = x + u * dt
        ubcs[i,:] = u
    return xbcs, ubcs

def solve_bc_ls(states, actions):
    #solve with least squares
    states = np.column_stack((states[:,0], states[:,1], np.ones(len(states))))

    theta, resid, rank, S = np.linalg.lstsq(states, actions, rcond=None)
    return theta


if __name__=="__main__":
    #plot a noisy p controller going to the origin

    x_inits = [np.array([-1,-1]), np.array([1.0, 1.0]), np.array([-1.0, 1.0]), np.array([1.0, -1.0])]
    k = 1.0
    dt = 0.05
    T = 500
    noise_std = 0.2
    rollouts = 20
    goal = np.array([0.0, 0.0])
    num_iter = 10
    lr = 0.0001#0.0005
    weight_decay = 0.00001
    #run bc off of one demo
    demo_start = x_inits[0]
    xs, us = get_demonstration(demo_start, k, dt, T, noise_std, goal)
    plt.plot(xs[:,0], xs[:,1],'o')
    plt.show()
    #run behavioral cloning...
    bc_controller = solve_bc_ls(xs, us)





    #
    #rollout bc_controller with noise
    epsilons = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
    colors = ['r','g','b','c','k','y']
    for n in range(len(epsilons)):
        for i in range(rollouts):
            xbcs, ubcs = get_bc_traj(bc_controller, demo_start, dt, T, epsilons[n])

            plt.plot(xbcs[:,0], xbcs[:,1],'o', color = colors[n])


    plt.show()
    ranked_bins = create_demo_bins(epsilons, bc_controller, demo_start, dt, T, num_rollouts = rollouts)
    #print("ranked_bins", ranked_bins)

    training_obs, training_labels = create_training_data_from_bins(ranked_bins)
    #print("training_obs", training_obs)
    #print("training_labels",training_labels)
    #learn reward net
    reward_net = Net()
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, training_obs, training_labels, optimizer, num_iter)
    torch.save(reward_net.state_dict(), "./point_mass_reward20.params")

    #print out the predicted returns for the noise injection
    for i in range(len(epsilons)):
        print("epsilon = {}".format(epsilons[i]))
        for t in ranked_bins[i]:
            traj_return = reward_net.forward(torch.from_numpy(t).float())
            print(traj_return)







    #test generalizatoin on several
    # for x_init in x_inits:
    #     #generate trajectory from bc
    #     xbcs, ubcs = get_bc_traj(bc_controller, x_init, dt, T)
    #
    #     #plot predicted over actual controls
    #     plt.plot(xs[:,0], xs[:,1],'bo')
    #     plt.plot(xbcs[:,0], xbcs[:,1],'ro')
    #
    #
    # plt.show()






    #test generalization of least squares solution...
