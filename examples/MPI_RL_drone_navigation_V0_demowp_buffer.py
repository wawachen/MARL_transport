#!/usr/bin/env python 
from os.path import dirname, join, abspath

import os
from pyrep.envs.bacterium_environment_mpi_RL import Drone_Env
from pyrep.common.arguments_v0 import get_args
from mpi4py import MPI
from pyrep.policies.maddpg_drone_att_demowp_buffer import MADDPG
# from pyrep.policies.maddpg_drone_scratch import MADDPG
from pyrep.common.replay_buffer_demo import Buffer
import numpy as np
from tensorboardX import SummaryWriter
import torch
import random
import math
from time import localtime, strftime
from scipy.io import savemat

DATA_DEMO = 0
DATA_RUNTIME = 1

def pretrain(args, buffer, agents, logger):
    print('Start pre training')
    for step in np.arange(args.pretrain_save_step, args.pretrain_step + 1, args.pretrain_save_step):
        if buffer.current_size >= args.batch_size:
            for agent in agents:
                agent.prep_training(device='cpu')
            for _ in range(args.pretrain_save_step):
                batch_data = buffer.sample(args.batch_size)
                for agent in agents:
                    other_agents = agents.copy()
                    other_agents.remove(agent)
                    agent.train(batch_data, other_agents, logger, args.use_gpu)
                   
        print("current pre step: ", step)
    
    for agent in agents:
        agent.prep_rollouts(device='cpu')
    print('Finish pre training')

def generate_actions(args,rank,state_list,policy_c,noise,epsilon,logger):
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
                action = policy.select_action(sc[i], noise, epsilon,args.use_gpu,logger) 
                actions.append(action)               
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

def initDemoBuffer(demoDataFile, Buffer,n_agent):
    demoData = np.load(demoDataFile)
    
    """episode_batch:  self.demo_episode x timestep x n x keydim   
    """
    action_episodes = demoData['acs']
    obs_episodes = demoData['obs']
    obs_next_episodes = demoData['obs_next']
    reward_episodes = demoData['r']

    assert(action_episodes.shape[0]==100)
    assert(action_episodes.shape[1]==60)

    if n_agent == 3:
        demo_timesteps = 25
    if n_agent == 4:
        demo_timesteps = 25
    if n_agent == 6:
        demo_timesteps = 35

    for i in range(action_episodes.shape[0]):
        for j in range(demo_timesteps):
            Buffer.store_episode(obs_episodes[i,j,:], action_episodes[i,j,:], reward_episodes[i,j,:], obs_next_episodes[i,j,:], DATA_DEMO)
    
    # print("Demo buffer size currently ", len(Buffer))

def run(args, env, agents,rank,comm):
    noise = args.noise_rate
    epsilon = args.epsilon
    episode_limit = args.max_episode_len
    max_episodes = args.max_episodes
    restart_frequency = 1000

    log_path = args.save_dir+"/log_drone_MPI_V0_demowp" 
    logger = SummaryWriter(logdir=log_path) # used for tensorboard
    
    i = 0
    
    if rank == 0:
        buffer = Buffer(args)
        if args.n_agents == 3:
            initDemoBuffer('orca_demonstration_ep100_3agents_env{}.npz'.format(args.field_size), buffer,3)
        if args.n_agents == 4:
            initDemoBuffer('orca_demonstration_ep100_4agents_env{}.npz'.format(args.field_size), buffer,4)
        if args.n_agents == 6:
            initDemoBuffer('orca_demonstration_ep100_6agents_env{}.npz'.format(args.field_size), buffer,6) 
        pretrain(args, buffer, agents, logger)

    while 1:
        if ((i%restart_frequency)==0)and(i!=0):
            env.restart()
        s = env.reset_world()

        if rank == 0:
            for agent in agents:
                agent.prep_rollouts(device='cpu')
        
        d_sg = 0
        reward = 0

        for t in range(episode_limit):
            state_list = comm.gather(s,root=0)
            #start = time.time()
            actions = generate_actions(args, rank,state_list,agents,noise,epsilon,logger)
            # print(actions)
            each_action = comm.scatter(actions,root=0)
            s_next, r, done, succ = env.step(each_action)

            reward += r[0]

            # add transitons in buff and update policy
            state_next_list = comm.gather(s_next, root=0) # n_env*[n,sdim]
            r_list = comm.gather(r, root=0)
            d_sg_list = comm.gather(d_sg, root=0)
        
            if rank == 0:
                for m in range(args.env_size):
                    if d_sg_list[m]==1:
                        pass  # do not store bad samples
                    else:
                        buffer.store_episode(state_list[m][:args.n_agents], actions[m], r_list[m][:args.n_agents], state_next_list[m][:args.n_agents], DATA_RUNTIME)

                if buffer.current_size >= args.batch_size:
                    for agent in agents:
                        agent.prep_training(device='cpu')
                    batch_data = buffer.sample(args.batch_size)
    
                    for agent in agents:
                        other_agents = agents.copy()
                        other_agents.remove(agent)
                        agent.train(batch_data, other_agents, logger, args.use_gpu)
                        agent.prep_rollouts(device='cpu')

            if d_sg==0:
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

        if rank==0:
            if i%10==0:
                for agent in agents:
                    agent.save_model(i)

        i+=1
        
        if rank==0:
            print("demo_episode{0}: {1}".format(i,reward))
        if i==max_episodes:
            break
            
    logger.close()
    env.shutdown()

def generate_evaluation_curve(args,env,agents):
    log_path = args.save_dir+"/log_drone_MPI_V0_demowp_buffer_evaluation" 
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    for agent in agents:
        agent.prep_rollouts(device='cpu')
    
    reward_all = []
    std_all = []

    for i in range(args.evaluate_episodes):
        for agent_num, agent in enumerate(agents):
            agent.initialise_networks(args.save_dir + '/' + 'agent_%d' % agent_num+ '/{0}_params.pkl'.format(i*10))
        
        returns = []
        for episode in range(3):
            # reset the environment
            s = env.reset_world()
            rewards = 0
            succ_list = np.zeros(args.evaluate_episode_len)
            
            for time_step in range(args.evaluate_episode_len):
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(agents):
                        action = agent.select_action_evaluate(torch.tensor(s[agent_id], dtype=torch.float32).unsqueeze(0), 0.05, 0.05, args.use_gpu, logger, 0) #sc[i], noise, epsilon,args.use_gpu,logger
                        actions.append(action[0,:])
                actions = np.array(actions)
                # print(actions.shape)
                s_next, r, done, succ = env.step_evaluate(actions)
                succ_list[time_step] = succ
                
                rewards += r[0]
                s = s_next

                if np.any(done):
                    break
                if np.sum(succ_list)==10:
                    break
                
            returns.append(rewards)

        reward_all.append(np.mean(returns))
        std_all.append(np.std(returns))
        logger.add_scalar('reward',np.mean(returns),i)
        logger.add_scalar('std', np.std(returns),i)
        print("demo_buffer_evaluation_episode{0}: {1}".format(i,np.mean(returns)))

    mdict = {'mean':reward_all,'std':std_all}
    savemat(args.save_dir,mdict)


def evaluation(args, env, agents):
    log_path = args.save_dir+"/log_drone_MPI_V0_demowp_evaluation_table"
    logger = SummaryWriter(logdir=log_path) # used for tensorboard
    returns = []
    time_f = []
    s_r = []
    deviation = []
    a_vel = []

    for agent in agents:
        agent.prep_rollouts(device='cpu')

    t_c = 0
    tt = 0
    t_c_ex = 0

    for episode in range(args.evaluate_episodes):
        # reset the environment
        s = env.reset_world()
        rewards = 0
        succ_list = np.zeros(args.evaluate_episode_len)
        pos_x = []
        pos_y = []
        
        vel_mag1 = []
        
        for time_step in range(args.evaluate_episode_len):
            vel_mag = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(agents):
                    action = agent.select_action_evaluate(torch.tensor(s[agent_id], dtype=torch.float32).unsqueeze(0), 0.05, 0.05, args.use_gpu, logger, t_c) #sc[i], noise, epsilon,args.use_gpu,logger
                    actions.append(action[0,:])
            actions = np.array(actions)
            # print(actions.shape)
            s_next, r, done, succ = env.step_evaluate(actions)
            succ_list[time_step] = succ
            for j in range(args.n_agents):
                logger.add_scalar('Agent%d/pos_x'%j, s_next[j,2]*args.field_size, t_c)
                logger.add_scalar('Agent%d/pos_y'%j, s_next[j,3]*args.field_size, t_c)
                logger.add_scalar('Agent%d/vel_x'%j, s_next[j,0], t_c)
                logger.add_scalar('Agent%d/vel_y'%j, s_next[j,1], t_c)
                pos_x.append(s_next[j,2]*args.field_size/2)
                pos_y.append(s_next[j,3]*args.field_size/2)    
                vel_mag.append(math.sqrt(s_next[j,0]**2+s_next[j,1]**2))

            vel_mag1.append(np.sum(np.array(vel_mag))/len(vel_mag))  
            
            rewards += r[0]
            s = s_next
            t_c += 1
            if np.any(done):
                break
            if np.sum(succ_list)==10:
                break
            
        logger.add_scalar('Success rate', np.any(succ_list), episode)
        logger.add_scalar('Finish time', t_c-1 ,episode)
        logger.add_scalar('Rewards', rewards, episode)
        returns.append(rewards)
        
        if np.any(succ_list):
            time_f.append(t_c-t_c_ex)
            a_vel.append(np.sum(np.array(vel_mag1))/len(vel_mag1))
            # tt += (t_c-t_c_ex)
        t_c_ex = t_c
        s_r.append(np.any(succ_list))

        if np.any(succ_list):
            pos_x_end = pos_x[-args.n_agents*10:]
            pos_y_end = pos_y[-args.n_agents*10:]
            if args.n_agents == 3:
                goals = np.array([[2.8,0],[-2.8,0],[0,0]])
            if args.n_agents == 4:
                goals = np.array([[1.25,1.25],[-1.25,1.25],[1.25,-1.25],[-1.25,-1.25]])
            if args.n_agents == 6:
                goals = np.array([[2.25,-2.25],[-2.25,-2.25],[1.2,0],[-1.2,0],[2.25,2.25],[-2.25,2.25]])
            
            dev_all = []
            for m in range(10):
                d_e_t =  0
                for i in range(args.n_agents):
                    d_e = np.zeros(args.n_agents)
                    for j in range(args.n_agents):
                        d_e[j] = np.sqrt(np.sum((goals[i,:]-np.array([pos_x_end[args.n_agents*m:args.n_agents*m+args.n_agents][j],pos_y_end[args.n_agents*m:args.n_agents*m+args.n_agents][j]]))**2))
                    d_e_t += np.min(d_e)
                d_e_ta = d_e_t/args.n_agents
                dev_all.append(d_e_ta)
            deviation.append(np.sum(np.array(dev_all))/10)

        print('Returns is', rewards)
    print("Results is")
    print("Finished time: ", np.sum(np.array(time_f))/len(time_f), ', ', np.std(np.array(time_f)) )
    print('Average speed: ', np.sum(np.array(a_vel))/len(a_vel), ', ', np.std(np.array(a_vel)))
    print("Success rate: ", np.sum(np.array(s_r))/len(s_r))
    print("Deviation: ", np.sum(np.array(deviation))/len(deviation),', ', np.std(np.array(deviation)))
    print("Total rewards", np.sum(np.array(returns)))
    logger.close()

    return sum(returns) / args.evaluate_episodes     

def set_global_seeds(seed):
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # get the params
    args = get_args()

    if args.field_size == 10:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')
    if args.field_size == 15:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_15x15.ttt')

    # create multiagent environment
    env = Drone_Env(args, env_name, args.n_agents)
    
    args.high_action = 1
    args.load_buffer = False
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    
    if not args.evaluate:
        args.save_dir = "./" + args.scenario_name + "/model_drone{}_demowp".format(args.n_agents)+'/'+'field_size{}'.format(args.field_size)+'/'+ 'env{}'.format(rank)+strftime("%Y-%m-%d--%H:%M:%S", localtime())
    else:
        args.save_dir = "./" + args.scenario_name + "/model_drone{}_demowp".format(args.n_agents)+'/'+'field_size{}'.format(args.field_size)+'/'+ 'env{}'.format(rank)
    
    args.use_gpu = False

    #buffer settings
    ######################################################
    args.seed = 222

    if args.n_agents == 3 or args.n_agents == 4:
        args.pretrain_save_step = 400 
        args.pretrain_step = 4000

    if args.n_agents == 6:
        args.pretrain_save_step = 600 
        args.pretrain_step = 6000
   
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
        if args.evaluate:
            if rank == 0:
                generate_evaluation_curve(args,env,agents)
        else:
            run(args, env, agents,rank,comm)
    except KeyboardInterrupt:
        env.shutdown()

