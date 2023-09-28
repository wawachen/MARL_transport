#!/usr/bin/env python 
from os.path import dirname, join, abspath

import os
from pyrep.envs.bacterium_environment_mpi_pc_global import Drone_Env
from pyrep.common.arguments import get_args
from mpi4py import MPI
from pyrep.policies.maddpg_drone_att_demowp_pc_global import MADDPG
# from pyrep.policies.maddpg_drone_scratch import MADDPG
from pyrep.common.replay_buffer_demowp import PrioritizedReplayBuffer
import numpy as np
from tensorboardX import SummaryWriter
import torch
from datetime import date
import random

DATA_DEMO = 0
DATA_RUNTIME = 1

def pretrain(args, buffer, agents, logger):
    print('Start pre training')
    for step in np.arange(args.pretrain_save_step, args.pretrain_step + 1, args.pretrain_save_step):
        if buffer.ready():
            for agent in agents:
                agent.prep_training(device='gpu')
            for _ in range(args.pretrain_save_step):
                batch_data, weights, idxes = buffer.sample(args.batch_size)
                p_list = np.zeros(args.batch_size)
                for agent in agents:
                    other_agents = agents.copy()
                    other_agents.remove(agent)
                    prio = agent.train(batch_data, other_agents, logger, args.use_gpu, weights)
                    p_list += np.array(prio)
                if not args.no_per:
                    p_list1 = p_list/args.n_agents
                    buffer.update_priorities(idxes, list(p_list1))
        print("current pre step: ", step)
    
    for agent in agents:
        agent.prep_rollouts(device='cpu')
    print('Finish pre training')

def initDemoBuffer(demoDataFile, Buffer):
    demoData = np.load(demoDataFile)
    
    """episode_batch:  self.demo_episode x timestep x n x keydim   
    """
    action_episodes = demoData['acs']
    obs_episodes = demoData['obs']
    obs_next_episodes = demoData['obs_next']
    reward_episodes = demoData['r']

    assert(action_episodes.shape[0]==100)
    assert(action_episodes.shape[1]==60)

    for i in range(action_episodes.shape[0]):
        for j in range(action_episodes.shape[1]):
            Buffer.add((obs_episodes[i,j,:], action_episodes[i,j,:], reward_episodes[i,j,:], obs_next_episodes[i,j,:], DATA_DEMO))

    Buffer.set_protect_size(len(Buffer))
    
    print("Demo buffer size currently ", len(Buffer))

def run(args, env, agents,rank,comm):
    noise = args.noise_rate
    epsilon = args.epsilon
    episode_limit = args.max_episode_len
    max_episodes = args.max_episodes
    restart_frequency = 1000
    buffer = PrioritizedReplayBuffer(args.buffer_size, args.seed, alpha=0.3, beta_init=1.0, beta_inc_n=100)

    log_path = os.getcwd()+"/log_drone3_MPI_pc1_global_demo_env{}".format(rank)
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    if args.n_agents == 3:
        initDemoBuffer('orca_demonstration_pc_ep100_3agents.npz', buffer)
    if args.n_agents == 6:
        initDemoBuffer('orca_demonstration_pc_ep100_6agents.npz', buffer) 
    pretrain(args, buffer, agents, logger)

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
            buffer.add((s[:args.n_agents], u, r[:args.n_agents], s_next[:args.n_agents],DATA_RUNTIME))
            s = s_next

            if buffer.ready():
                for agent in agents:
                    agent.prep_training(device='gpu')
                batch_data, weights, idxes = buffer.sample(args.batch_size)
                p_list = np.zeros(args.batch_size)
                for agent in agents:
                    other_agents = agents.copy()
                    other_agents.remove(agent)
                    prio = agent.train(batch_data, other_agents, logger, args.use_gpu, weights)
                    p_list += np.array(prio)
                    agent.prep_rollouts(device='cpu')

                if not args.no_per:
                    p_list1 = p_list/args.n_agents
                    buffer.update_priorities(idxes, list(p_list1))

            noise = max(0.05, noise - 0.0000005)
            epsilon = max(0.05, noise - 0.0000005)

            if np.any(done):
                break

        logger.add_scalar('mean_episode_pc1_global_rewards{}'.format(rank), score, i)
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
    num_agents = 3

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
    args.save_dir = "./model_drone3_mpi_pc1_global_demo_train{}".format(rank)
    args.env_size = 1 #initial game number
    args.use_gpu = False

    #buffer settings
    args.seed = 54760
    if num_agents == 3:
        args.pretrain_save_step = 400 
        args.pretrain_step = 4000

    if num_agents == 6:
        args.pretrain_save_step = 600 
        args.pretrain_step = 6000
        # args.noise_rate = 0.15

    args.no_per = False
    args.const_demo_priority = 0.99
    args.const_min_priority =  0.001

    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []        
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2) 
    args.stage = 1

    ##########################
    # args.seed = None
    # if args.seed is None:
    #     args.seed = random.randint(0,10000)
    
    # torch.manual_seed(args.seed)
    # # torch.set_num_threads(1)
    # np.random.seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    ##########################

    agents = [MADDPG(args,i,rank) for i in range(args.n_agents)]

    try:
        run(args, env, agents,rank,comm)
    except KeyboardInterrupt:
        env.shutdown()

