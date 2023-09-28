from os import path
from os.path import dirname, join, abspath

from pyrep.envs.orca_bacterium_environment_demo_global import Drone_Env
from scipy.io import savemat
import signal
import numpy as np

from os.path import dirname, join, abspath

import os
# from pyrep.policies.dqn import DQNAgent
from pyrep.common.arguments import get_args
from pyrep.common.rollout_drone_navigation_demo_global import Rollout

if __name__ == '__main__':
    # get the params
    args = get_args()
    num_agents = 3

    if num_agents==3:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_square_six_s.ttt')
        
    if num_agents==6:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_square_six.ttt')
       
    if num_agents==12:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_square_twe.ttt')

    # create multiagent environment
    env = Drone_Env(env_name,num_agents)
 
    args.high_action = 1 
    args.max_episodes = 100 #8000
    args.max_episode_len = 60 #60
    args.n_agents = num_agents # agent number in a swarm
    args.evaluate_rate = 10000 
    args.evaluate = False #
    args.load_buffer = False
    args.evaluate_episode_len = 30
    args.evaluate_episodes = 20
    args.save_rate = 1000
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    args.save_dir = "./model_drone6_stratch"
    args.scenario_name = "navigation_drone6_stratch"
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []        
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2) 

    rollout = Rollout(args, env)

    if args.evaluate:
        returns = rollout.evaluate()
        print('Average returns is', returns)
    else:
        observations,actions,observations_next,rewards = rollout.run()
        fileName = "orca_demonstration_pc"

        fileName += "_ep" + str(args.max_episodes)

        fileName += "_" + str(args.n_agents) + "agents"

        fileName += ".npz"
    
        np.savez_compressed(fileName, acs=actions, obs=observations, obs_next=observations_next,r=rewards)
        print("finish saving file")
    
    env.shutdown()









