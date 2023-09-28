#!/usr/bin/env python 
from os.path import dirname, join, abspath

import os
from pyrep.envs.bacterium_environment_mpi_pc import Drone_Env
from pyrep.common.arguments import get_args
from mpi4py import MPI
from pyrep.policies.maddpg_drone_att_pc1 import MADDPG
# from pyrep.policies.maddpg_drone_scratch import MADDPG
from pyrep.common.replay_buffer import Buffer
import numpy as np
from tensorboardX import SummaryWriter
import torch
from datetime import date

def run(args, env, agents,rank,comm):
    noise = args.noise_rate
    epsilon = args.epsilon
    episode_limit = args.max_episode_len
    max_episodes = args.max_episodes
    restart_frequency = 1000
    buffer = Buffer(args)

    log_path = os.getcwd()+"/log_drone6_MPI_pc2_env{}".format(rank)
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    for i in range(max_episodes):
        if ((i%restart_frequency)==0)and(i!=0):
            env.restart()
        s = env.reset_world()
        score = 0
       
        for agent in agents:
            agent.prep_rollouts(device='cpu')

        for t in range(episode_limit):
            ###########
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(agents):                       
                    action = agent.select_action(s[agent_id], noise, epsilon, args.use_gpu, logger)
                    u.append(action)
                    actions.append(action)
            ###########
            s_next, r, done = env.step(actions)
            score += r[0]

            ###########
            buffer.store_episode(s[:args.n_agents], u, r[:args.n_agents], s_next[:args.n_agents])
            s = s_next

            if buffer.current_size >= args.batch_size:
                for agent in agents:
                    agent.prep_training(device='cpu')
                transitions = buffer.sample(args.batch_size)
                for agent in agents:
                    other_agents = agents.copy()
                    other_agents.remove(agent)
                    agent.train(transitions, other_agents,logger,args.use_gpu)
                    agent.prep_rollouts(device='cpu')

            noise = max(0.05, noise - 0.0000005)
            epsilon = max(0.05, noise - 0.0000005)

            if np.any(done):
                break

        logger.add_scalar('mean_episode_pc2_rewards{}'.format(rank), score, i)
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

    env_name = join(dirname(abspath(__file__)), 'RL_drone_square_six.ttt')

    # create multiagent environment
    env = Drone_Env(env_name,num_agents)
    
    args.selection_num = 3
    args.high_action = 1
    args.max_episodes = 3000
    args.max_episode_len = 80 #60
    args.n_agents = num_agents # agent number in a swarm
    args.evaluate_rate = 10000 
    args.load_buffer = False
    # args.evaluate_episode_len = 30
    # args.evaluate_episodes = 20
    args.save_rate = 1000
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space

    args.env_size = args.selection_num*(args.selection_num+1)/2 #initial game number
    args.use_gpu = False
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []        
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2) 

    args.stage = 3

    with open('index.npy', 'rb') as f:
        index = np.load(f)

    save_name_combo = []
    load_name_combo1 = []
    load_name_combo2 = []
    index_r = []

    for i in range(args.selection_num):
        for j in range(i,args.selection_num):
            save_name_combo.append("./model_drone6_pc2_train{}-{}".format(index[i],index[j]))
            load_dir1 = "./model_drone3_mpi_pc1_train{}".format(index[i])
            load_dir2 = "./model_drone3_mpi_pc1_train{}".format(index[j])
            load_name_combo1.append(load_dir1)
            load_name_combo2.append(load_dir2)
            index_r.append([index[i],index[j]])

    args.save_dir = save_name_combo[rank]
    args.load_dir1 = load_name_combo1[rank]
    args.load_dir2 = load_name_combo2[rank]
    args.index_r = index_r[rank]

    agents = [MADDPG(args,i,rank) for i in range(args.n_agents)]

    try:
        run(args, env, agents,rank,comm)
    except KeyboardInterrupt:
        env.shutdown()

