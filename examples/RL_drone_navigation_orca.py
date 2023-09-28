#!/usr/bin/env python 
from os.path import dirname, join, abspath

import os
from pyrep.envs.bacterium_environment_orca import Drone_Env
# from pyrep.policies.dqn import DQNAgent
from pyrep.common.arguments import get_args
from pyrep.common.rollout_drone_navigation_orca import Rollout

if __name__ == '__main__':
    # get the params
    args = get_args()

    env_name = join(dirname(abspath(__file__)), 'RL_drone_square.ttt')

    num_agents = 12
    # create multiagent environment
    env = Drone_Env(env_name,num_agents)
 
    args.high_action = 1
    args.max_episodes = 8000 #8000
    args.max_episode_len = 80 #60
    args.n_agents = num_agents # agent number in a swarm
    args.evaluate_rate = 10000 
    args.evaluate = False #
    args.load_buffer = False
    args.evaluate_episode_len = 30
    args.evaluate_episodes = 20
    args.save_rate = 1000
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    args.save_dir = "./model_drone12_orca"
    args.scenario_name = "navigation_drone12_orca"
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
        rollout.run()
    
    env.shutdown()

