import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from point_mass import Net


reward_net = Net()
reward_net.load_state_dict(torch.load("./point_mass_reward20.params"))

#plot out the reward functions
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter


with torch.no_grad():
    pv_returns = np.array([[reward_net.forward(torch.from_numpy(np.array([[x,y]])).float()).item() for x in np.linspace(-2,2,100)]
                           for y in np.linspace(2,-2,100)])

#print(np.array([[[x,y] for x in np.linspace(-2,2,100)]
#                       for y in np.linspace(2,-2,100)]))

#print(pv_returns)
#uniform_data = np.array([[0,1,2],[4,3,2],[-1,-2,-10]])
#ax = sns.heatmap(uniform_data)
ax = sns.heatmap(pv_returns)

print(pv_returns.shape)
#ax.set_xticks(np.linspace(-2,2,5))
#ax.set_yticks(np.linspace(2,-2,5))
#ax.set_xticklabels([round(float(label), 2) for label in np.linspace(-2,2)])
#ax.set_yticklabels([round(float(label), 2) for label in np.linspace(2,-2)])
plt.xlabel("x-position")
plt.ylabel("y-position")
#plt.savefig("learned_mcar_return.png")
plt.show()


# traj_return = reward_net.forward(torch.from_numpy(xs).float())
# print(traj_return)


# colors = ['r','g','b','c','k','y']
# for n in range(len(epsilons)):
#     for i in range(20):
#         xbcs, ubcs = get_bc_traj(bc_controller, demo_start, dt, T, epsilons[n])
#
#         plt.plot(xbcs[:,0], xbcs[:,1],'o', color = colors[n])
#
#
# plt.show()



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
