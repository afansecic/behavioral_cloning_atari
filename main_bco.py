import torch
import argparse
import numpy as np
from train import train, train_transitions
from pdb import set_trace
import dataset
import tensorflow as tf
from run_test import *
from torch.autograd import Variable
import utils




def mask_score(obs):
    #takes a stack of four observations and blacks out (sets to zero) top n rows
    n = 10
    #no_score_obs = copy.deepcopy(obs)
    obs[:,:n,:,:] = 0
    return obs

def generate_transitions(env, num_steps = 100000):
    print("learning inverse transition dynamics")
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
    env = make_vec_env(env_id, env_type, 1, 0, wrapper_kwargs={'clip_rewards':False,'episode_life':False,})
    if env_type == 'atari':
        env = VecFrameStack(env, 4)


    #run steps in env and learn transitions
    step_cnt = 0
    transitions = []
    while step_cnt < num_steps:
        ob = env.reset()
        state = mask_score(ob)
        state = np.transpose(state, (0, 3, 1, 2)).squeeze()

        while True:
            if step_cnt % 1000 == 0:
                print("generated {} state transitions".format(step_cnt))
            #preprocess the state
            action = env.action_space.sample()
            ob, reward, done, _ = env.step(action)
            next_state = mask_score(ob)
            next_state = np.transpose(next_state, (0, 3, 1, 2)).squeeze()
            #print(next_state.shape)
            #print(state.shape)
            stacked = np.concatenate((state, next_state), axis=0)
            #print(stacked.shape)
            transitions.append((stacked, action))
            state = next_state
            step_cnt += 1
            if step_cnt >= num_steps:
                break
            if done:
                #print("done")
                break
    return transitions


def generate_novice_demo_observations(env, env_name, agent):
    checkpoint_min = 50
    checkpoint_max = 600
    checkpoint_step = 50
    checkpoints = []
    if env_name == "enduro":
        checkpoint_min = 3100
        checkpoint_max = 3650
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints.append('0000')
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    print(checkpoints)



    demonstrations = []
    learning_returns = []
    learning_rewards = []
    for checkpoint in checkpoints:

        model_path = "../learning-rewards-of-learners/learner/models/" + env_name + "_25/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            traj = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            #traj.append(ob)
            #print(ob.shape)
            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)
                traj.append(mask_score(ob))
                ob, r, done, _ = env.step(action)
                #print(ob.shape)

                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    break
            print("traj length", len(traj))
            print("demo length", len(demonstrations))
            demonstrations.append(traj)
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)
    print(np.mean(learning_returns), np.max(learning_returns))
    return demonstrations, learning_returns, learning_rewards



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ##################################################
    # ##             Algorithm parameters             ##
    # ##################################################
    #parser.add_argument("--dataset-size", type=int, default=75000)
    #parser.add_argument("--updates", type=int, default=10000)#200000)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--hist-len", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--env_name", type=str, help="Atari environment name in lowercase, i.e. 'beamrider'")

    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--min-squared-gradient", type=float, default=0.01)
    parser.add_argument("--l2-penalty", type=float, default=0.0001)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_bco")
    parser.add_argument("--num_eval_episodes", type=int, default = 30)
    parser.add_argument('--seed', default=0, help="random seed for experiments")

    args = parser.parse_args()
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
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    stochastic = True


    #load novice demonstrations
    #pkl_file = open("../learning-rewards-of-learners/learner/novice_demos/" + args.env_name + "12_50.pkl", "rb")
    #novice_data = pickle.load(pkl_file)
    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    #run self play for inverse transition dynamics learning
    transitions = generate_transitions(env_name)

    transition_dataset_size = len(transitions)
    print("Transition Data set size = ", transition_dataset_size)

    transition_data = dataset.Dataset(transition_dataset_size, args.hist_len*2)
    num_data = 0
    action_set = set()
    action_cnt_dict = {}
    for ssa in transitions:
        states, action = ssa
        action = action
        action_set.add(action)
        if action in action_cnt_dict:
            action_cnt_dict[action] += 1
        else:
            action_cnt_dict[action] = 0
        #transpose into 4x84x84 format
        transition_data.add_item(states, action)
        num_data += 1
        if num_data == transition_dataset_size:
            print("data set full")
            break
    print("available actions", action_set)
    print(action_cnt_dict)

    transition_model = train_transitions(args.env_name,
        action_set,
        args.learning_rate,
        args.alpha,
        args.min_squared_gradient,
        args.l2_penalty,
        args.minibatch_size,
        args.hist_len*2,
        args.discount,
        args.checkpoint_dir,
        transition_dataset_size*4,
        transition_data, args.num_eval_episodes)


    demonstrations, learning_returns, _ = generate_novice_demo_observations(env, env_name, agent)

    #TODO:
    #classify the actions of the demonstrations
    #take each consecutive state, next state and concatentate
    #run transition_model.get_action(state) to get action
    #add action
    demonstrations_pred = []
    for d in demonstrations:
        for i in range(len(d)-1):
            #add state i and state i + 1 to predict action
            state = np.transpose(np.squeeze(d[i]), (2, 0, 1))
            next_state = np.transpose(np.squeeze(d[i+1]), (2, 0, 1))
            transition = np.concatenate((state, next_state), axis=0)
            #print(transition.shape)
            transition = Variable(utils.float_tensor(transition))
            #print(transition.size())
            with torch.no_grad():
                action_pred = transition_model.get_action(transition.unsqueeze(0))
                #print(action_pred)
            demonstrations_pred.append((state, action_pred))


    dataset_size = len(demonstrations_pred)
    print("Data set size = ", dataset_size)

    data = dataset.Dataset(dataset_size, args.hist_len)
    episode_index_counter = 0
    num_data = 0
    for sa in demonstrations_pred:
        state, action = sa
        #print(action)
        #input("enter")
        action_set.add(action)
        #transpose into 4x84x84 format
        data.add_item(state, action)
        num_data += 1
        if num_data == dataset_size:
            print("data set full")
            break

    print("available actions", action_set)
    print(action_cnt_dict)

    train(args.env_name,
        action_set,
        args.learning_rate,
        args.alpha,
        args.min_squared_gradient,
        args.l2_penalty,
        args.minibatch_size,
        args.hist_len,
        args.discount,
        args.checkpoint_dir,
        dataset_size*3,
        data, args.num_eval_episodes)
