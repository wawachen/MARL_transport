#!/usr/bin/env python 
from os.path import dirname, join, abspath

import os
from pyrep.envs.bacterium_environment_mpi import Drone_Env
from pyrep.common.arguments import get_args
from mpi4py import MPI
# from pyrep.policies.maddpg_drone_att import MADDPG
from pyrep.policies.maddpg_drone_scratch import MADDPG
from pyrep.common.replay_buffer import Buffer
import numpy as np
from tensorboardX import SummaryWriter
import torch

def generate_actions(rank,state_list,policy_c,noise,epsilon):
    #state_list n_env*[n,sdim]---> n_env*[n,adim]
    if rank == 0:
        sc = [[] for _ in range(state_list[0].shape[0])]
        for _,sl in enumerate(state_list):
            for j in range(state_list[0].shape[0]):           
                sc[j].append(sl[j,:])
        sc = [torch.tensor(sci,dtype=torch.float32) for sci in sc] #n*[n_env,sdim]
        with torch.no_grad():
            actions = []
            for i,policy in enumerate(policy_c): 
                actions.append(policy.select_action(sc[i], noise, epsilon))               
        #n*[n_env,adim]
        # print(actions)
        ac = [[] for _ in range(len(state_list))]
        for _,al in enumerate(actions):
            for j in range(actions[0].shape[0]): 
                # print(al.shape)          
                ac[j].append(al[j,:])
        ac = [np.array(aci) for aci in ac] #n_env*[n,adim]
    else:
        ac = None
    return ac

def run(args, env, agents,rank,comm):
    noise = args.noise_rate
    epsilon = args.epsilon
    episode_limit = args.max_episode_len
    max_episodes = args.max_episodes
    restart_frequency = 1000
    buffer = Buffer(args)
    log_path = os.getcwd()+"/log_drone3_MPI_d"
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    for i in range(max_episodes):
        if ((i%restart_frequency)==0)and(i!=0):
            env.restart()
        s = env.reset_world()
        score = 0
        # if rank == 0:
        #     for agent in agents:
        #         agent.prep_rollouts(device='cpu')
        if rank:
            d_sg = 1
        else:
            d_sg = 0

        for t in range(episode_limit):
            state_list = comm.gather(s,root=0)
            #start = time.time()
            actions = generate_actions(rank,state_list,agents,noise,epsilon)
            each_action = comm.scatter(actions,root=0)
            s_next, r, done = env.step(each_action)
            score += r[0]

            # add transitons in buff and update policy
            state_next_list = comm.gather(s_next, root=0) # n_env*[n,sdim]
            r_list = comm.gather(r, root=0)
            d_sg_list = comm.gather(d_sg, root=0)
        
            if rank == 0:
                for m in range(args.env_size):
                    if d_sg_list[m]==1:
                        pass  # do not store bad samples
                    else:
                        buffer.store_episode(state_list[m][:args.n_agents],actions[m], r_list[m][:args.n_agents], state_next_list[m][:args.n_agents])

                if buffer.current_size >= args.batch_size:
                    # if args.use_gpu:
                    #     for agent in agents:
                    #         agent.prep_training(device='gpu')
                    # else:
                    #     for agent in agents:
                    #         agent.prep_training(device='cpu')
                    transitions = buffer.sample(args.batch_size)
                    for agent in agents:
                        other_agents = agents.copy()
                        other_agents.remove(agent)
                        agent.train(transitions,other_agents,logger,args.use_gpu)
                        # agent.prep_rollouts(device='cpu')

            noise = max(0.05, noise - 0.0000005)
            epsilon = max(0.05, noise - 0.0000005)

            s = s_next

            if rank==0:
                done_0 = [done]*args.env_size
            else:
                done_0 = None

            d_0 = comm.scatter(done_0,root=0)

            
            if np.any(d_0):
                break

            if rank:
                if np.any(done) or d_sg:
                    d_sg = 1
               

        logger.add_scalar('mean_episode_rewards{}'.format(rank), score, i)
        if rank==0:
            print("episode",i,":",score)
        # logger.add_scalar('network_loss', loss_sum, i)
            
    logger.close()
    env.shutdown()
        

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # get the params
    args = get_args()

    env_name = join(dirname(abspath(__file__)), 'RL_drone_square{}.ttt'.format(rank))
    num_agents = 3
    # create multiagent environment
    env = Drone_Env(env_name,num_agents)
    
    args.high_action = 1
    args.max_episodes = 8000 #8000
    args.max_episode_len = 60 #60
    args.n_agents = num_agents # agent number in a swarm
    args.evaluate_rate = 10000 
    args.evaluate = False #
    args.load_buffer = False
    args.evaluate_episode_len = 30
    args.evaluate_episodes = 20
    args.save_rate = 1000
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    args.save_dir = "./model_drone3_mpi_d"
    args.scenario_name = "navigation_drone3_mpi_d"
    args.env_size = 2
    args.use_gpu = False
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []        
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2) 

    if rank==0:
        agents = [MADDPG(args,i) for i in range(args.n_agents)]
    else:
        agents = None

    try:
        run(args, env, agents,rank,comm)
    except KeyboardInterrupt:
        env.shutdown()

