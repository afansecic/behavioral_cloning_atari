import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from point_mass import Net

#I want to run ES to get a good controller that generalizes

def evaluate_traj(traj, reward_net):
    #run traj through reward net
    traj_return = reward_net.forward(torch.from_numpy(traj).float())
    return traj_return

def evaluate_controller(reward_net, controller, start_states, dt, T, k):
    returns = []
    #assume controller is 3x3 matrix like BC for now
    for s in start_states:
        xs, us = get_bc_traj(controller, s, dt, T, 0.0, k)
        returns.append(evaluate_traj(xs, reward_net).item())
    return np.mean(returns)

def policy_optimize_es(num_trials, reward_net, start_states, dt, T, k):
    #use evoluationary search for a controller
    #greedy search for now
    best_perf = -np.float('inf')
    best_controller = 2 * np.random.rand(3,2) - 1
    for i in range(num_trials):
        controller = best_controller + 2*np.random.randn(3,2)
        controller[controller > 6.0] = 6.0
        controller[controller < -6.0] = -6.0
        #print(controller)
        #input()
        controller_perf = evaluate_controller(reward_net, controller, start_states, dt, T, k)
        if controller_perf > best_perf:
            print(i)
            best_perf = controller_perf
            best_controller = controller
            print(best_perf)
            print(best_controller)
    return best_controller, best_perf

def policy_optimize_random(num_trials, reward_net, start_states, dt, T, k):
    #use evoluationary search for a controller
    #greedy search for now
    best_perf = -np.float('inf')
    best_controller = 2 * np.random.rand(3,2) - 1
    for i in range(num_trials):
        controller = 4.0 * np.random.rand(3,2) - 2.0
        controller_perf = evaluate_controller(reward_net, controller, start_states, dt, T, k)
        if controller_perf > best_perf:
            print(i)
            best_perf = controller_perf
            best_controller = controller
            print(best_perf)
            print(best_controller)
    return best_controller, best_perf

def get_bc_traj(bc_controller, init_x, dt, T, eps_greedy, eps_k):

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
            #print(bc_controller)
            u = np.dot(np.append(x,[1.0]),bc_controller)
            #print(np.append(x,[1.0]))
            #print(u)
            #input()
        #print("u", u)
        x = x + u * dt
        ubcs[i,:] = u
    return xbcs, ubcs



if __name__=="__main__":
    #plot a noisy p controller going to the origin

    x_inits = [np.array([-1,-1]), np.array([1.0, 1.0]), np.array([-1.0, 1.0]), np.array([1.0, -1.0])]
    k = 1.0
    dt = 0.05
    T = 500
    eps_greedy = 0.0
    trials = 2000
    goal = np.array([0.0, 0.0])

    reward_net = Net()
    reward_net.load_state_dict(torch.load("./point_mass_reward20.params"))

    controller_es, controller_perf = policy_optimize_es(trials,reward_net, x_inits, dt, T, k)
    print("perf", controller_perf)
    print(controller_es)
    #test generalizatoin on several
    for x_init in x_inits:
        #generate trajectory from bc
        xbcs, ubcs = get_bc_traj(controller_es, x_init, dt, T, eps_greedy, k)

        #plot predicted over actual controls
        #plt.plot(xs[:,0], xs[:,1],'bo')
        plt.plot(xbcs[:,0], xbcs[:,1],'ro')


    plt.show()






    #test generalization of least squares solution...
