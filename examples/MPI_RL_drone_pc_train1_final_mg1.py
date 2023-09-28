#!/usr/bin/env python 
from os.path import dirname, join, abspath

import os
from pyrep.envs.bacterium_environment_mpi_pc_global import Drone_Env
from pyrep.common.arguments import get_args
from mpi4py import MPI
from pyrep.policies.maddpg_drone_att_pc1_final_mg import MADDPG
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

    log_path = os.getcwd()+"/log_drone3_MPI_pc1_env{}".format(rank)
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
                # actions = agent.select_actions(s, 0.05, 0.05, args.use_gpu, logger)
               
            ###########
            s_next, r, done = env.step(actions)
            score += r[0]

            ###########
            buffer.store_episode(s[:args.n_agents], actions, r[:args.n_agents], s_next[:args.n_agents])
            s = s_next

            if buffer.current_size >= args.batch_size and i>=500:
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
    num_agents = 6

    env_name = join(dirname(abspath(__file__)), 'RL_drone_square_six_s.ttt')

    # create multiagent environment
    env = Drone_Env(env_name,num_agents,rank)
    
    args.high_action = 1
    args.max_episodes = 5000
    args.max_episode_len = 80 #
    args.n_agents = num_agents # agent number in a swarm
    args.evaluate_rate = 10000 
    args.evaluate = False #
    args.load_buffer = False
    # args.evaluate_episode_len = 30
    # args.evaluate_episodes = 20
    args.save_rate = 1000
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    args.save_dir = "./model_drone6_mpi_d1_cur2"
    args.env_size = 1 #initial game number
    args.use_gpu = False
    args.lr_actor = 0.01
    args.lr_critic = 0.01
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []        
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2) 
    args.stage = 1

    agent = MADDPG(args,rank) 

    try:
        run(args, env, agent,rank,comm)
    except KeyboardInterrupt:
        env.shutdown()

