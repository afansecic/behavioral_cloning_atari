import argparse
import sys

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

#from mujoco_py import GlfwContext
#GlfwContext(offscreen=True)  # Create a window to init GLFW.

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

    def act(self, observation, reward, done):
        if self.stochastic:
            a,v,state,neglogp = self.model.step(observation)
        else:
            a = self.model.act_model.act(observation)
        return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--episode_count', type=int, default=100)
    parser.add_argument('--stochastic', action='store_true')
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



    env = VecNormalize(env,ob=True,ret=False,eval=True)

    try:
        env.load(args.model_path) # Reload running mean & rewards if available
    except AttributeError:
        pass

    agent = PPO2Agent(env,args.env_type,args.stochastic)
    agent.load(args.model_path)
    #agent = RandomAgent(env.action_space)

    episode_count = args.episode_count
    reward = 0
    done = False

    #record the data for behavioral cloning
    demonstrations  = []
    for i in range(episode_count):
        ob = env.reset()
        trajectory_actions = []
        trajectory_observations = []
        steps = 0
        acc_reward = 0
        while True:
            #get grayscale image
            rgb_img = env.render('rgb_array')
            gray_scale = color.rgb2gray(rgb_img)
            action = agent.act(ob, reward, done)

            trajectory_observations.append(gray_scale)
            trajectory_actions.append(action)

            ob, reward, done, _ = env.step(action)

            steps += 1
            acc_reward += reward

            if done:
                print(steps,acc_reward)
                break
        demonstrations.append((trajectory_observations, trajectory_actions))
        # plt.imshow(rgb_img)
        # plt.show()

    env.close()
    env.venv.close()
