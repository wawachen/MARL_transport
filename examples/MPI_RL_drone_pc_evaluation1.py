#!/usr/bin/env python 
from os.path import dirname, join, abspath

import os
from pyrep.envs.bacterium_environment_mpi_pc import Drone_Env
from pyrep.common.arguments import get_args
from mpi4py import MPI
from pyrep.policies.maddpg_drone_att_pc1 import MADDPG
# from pyrep.policies.maddpg_drone_scratch import MADDPG
import numpy as np
from tensorboardX import SummaryWriter
import torch

def evaluation(args, env, agents, rank):
    log_path = os.getcwd()+"/log_drone3_MPI_pc1_eval{}".format(rank)
    logger = SummaryWriter(logdir=log_path) # used for tensorboard
    returns = []
    for episode in range(args.evaluate_episodes):
        # reset the environment
        s = env.reset_world()
        rewards = 0
        for agent in agents:
            agent.prep_rollouts(device='cpu')

        for time_step in range(args.evaluate_episode_len):
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(agents):
                    action = agent.select_action(s[agent_id], 0.05, 0.05, args.use_gpu, logger) #sc[i], noise, epsilon,args.use_gpu,logger
                    actions.append(action)

            # print(actions.shape)
            s_next, r, done = env.step(actions)
            rewards += r[0]
            s = s_next
            if np.any(done):
                break
        returns.append(rewards)
        print('Returns is', rewards)

    logger.close()
    env.shutdown()

    return sum(returns) / args.evaluate_episodes  
        

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # get the params
    args = get_args()
    num_agents = 3

    env_name = join(dirname(abspath(__file__)), 'RL_drone_square_six.ttt')

    # create multiagent environment
    env = Drone_Env(env_name,num_agents)
    
    args.high_action = 1
    args.selection_num = 3
    # args.max_episodes = 2000 #8000
    # args.max_episode_len = 80 #60
    args.n_agents = num_agents # agent number in a swarm
    args.evaluate_rate = 10000 
    args.load_buffer = False
    args.evaluate_episode_len = 80 #80
    args.evaluate_episodes = 200
    args.save_rate = 1000
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    args.load_dir = "./model_drone3_mpi_pc1_train{}".format(rank) # needs to be edited
    args.save_dir = "./model_drone3_mpi_pc1_evaluation{}".format(rank)
    args.env_size = 6 #initial game number
    args.use_gpu = False
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []        
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2) 
    args.stage = 2

    agents = [MADDPG(args,i,rank) for i in range(args.n_agents)]

    try:
        reward_average = evaluation(args, env, agents,rank)
        sum_rewards = comm.gather(reward_average,root=0)
        
        if rank == 0:
            sum_rewards = np.array(sum_rewards)
            index = np.argsort(sum_rewards)[-args.selection_num:]

            with open('index.npy', 'wb') as f:
                np.save(f, index)

            print("index: ", index)

    except KeyboardInterrupt:
        env.shutdown()

