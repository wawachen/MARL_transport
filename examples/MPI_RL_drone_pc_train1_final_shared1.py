#!/usr/bin/env python 
from os.path import dirname, join, abspath

import os
from pyrep.envs.bacterium_environment_mpi_pc_global_5m1 import Drone_Env
from pyrep.common.arguments import get_args
from mpi4py import MPI
from pyrep.policies.maddpg_drone_att_pc1_shared1 import MADDPG
# from pyrep.policies.maddpg_drone_scratch import MADDPG
from pyrep.common.replay_buffer import Buffer
import numpy as np
from tensorboardX import SummaryWriter
import torch
from datetime import date

def run(args, env, agent,rank,comm):
    noise = args.noise_rate
    epsilon = args.epsilon
    episode_limit = args.max_episode_len
    max_episodes = args.max_episodes
    restart_frequency = 1000
    buffer = Buffer(args)

    log_path = os.getcwd()+"/log_drone2_MPI_pc1_mini_env{}".format(rank)
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    for i in range(max_episodes):
        if ((i%restart_frequency)==0)and(i!=0):
            env.restart()
    
        s = env.reset_world()
        score = 0
        
        agent.prep_rollouts(device='cpu')

        for t in range(episode_limit):
            ###########
            u = []
            with torch.no_grad():                    
                actions = agent.select_actions(s, noise, epsilon, args.use_gpu, logger)
               
            ###########
            s_next, r, done = env.step(actions)
            score += r[0]

            ###########
            buffer.store_episode(s[:args.n_agents], actions, r[:args.n_agents], s_next[:args.n_agents])
            s = s_next

            if buffer.current_size >= args.batch_size:
                agent.prep_training(device='cpu')
                transitions = buffer.sample(args.batch_size)
               
                agent.train(transitions, logger,args.use_gpu)
                agent.prep_rollouts(device='cpu')

            noise = max(0.05, noise - 0.0000005)
            epsilon = max(0.05, noise - 0.0000005)

            if np.any(done):
                break

        logger.add_scalar('mean_episode_pc1_rewards{}'.format(rank), score, i)
        # logger.add_scalar('network_loss', loss_sum, i)
    
        print("Env%d_episode%d"%(rank,i),":",score)
            ###########
            
    logger.close()
    env.shutdown()
        

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # get the params
    args = get_args()
    num_agents = 2

    env_name = join(dirname(abspath(__file__)), 'RL_drone_square_six_ss.ttt')

    # create multiagent environment
    env = Drone_Env(env_name,num_agents,rank)
    
    args.high_action = 1
    args.max_episodes = 10000
    args.max_episode_len = 80 #
    args.n_agents = num_agents # agent number in a swarm
    args.evaluate_rate = 10000 
    args.evaluate = False #
    args.load_buffer = False
    # args.evaluate_episode_len = 30
    # args.evaluate_episodes = 20
    args.stage = 1
    args.save_rate = 1000
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    # args.save_dir = "./model_drone2_mpi_pc1_final_shared_mini1_train{}".format(rank)
    args.save_dir = "./" + "Navigation_curriculum2" + "/stage{}".format(args.stage) +"/model_drone{}".format(args.n_agents)
    args.env_size = 1 #initial game number
    args.use_gpu = False
    # args.batch_size = 102 ####edit!!
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []        
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2) 

    agent = MADDPG(args,rank) 

    try:
        run(args, env, agent,rank,comm)
    except KeyboardInterrupt:
        env.shutdown()

