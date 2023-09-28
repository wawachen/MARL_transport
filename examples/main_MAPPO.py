#!/usr/bin/env python 
from os.path import dirname, join, abspath

import os
from pyrep.envs.bacterium_environment_mpi_RL_mappo import Drone_Env
from pyrep.common.arguments_mappo import get_args
# from pyrep.policies.maddpg_drone_scratch import MADDPG
import numpy as np
from tensorboardX import SummaryWriter
import torch
import math
from time import localtime, strftime
import random

from pyrep.baselines.mappo.normalization import Normalization, RewardScaling
from pyrep.baselines.mappo.replay_buffer import ReplayBuffer
from pyrep.baselines.mappo.mappo_mpe import MAPPO_MPE

def run(args, env, agent_n):
    episode_limit = args.episode_limit
    restart_frequency = 1000
    replay_buffer = ReplayBuffer(args)
    log_path = args.save_dir+"/log_drone_MPI_V0" 
    logger = SummaryWriter(logdir=log_path) # used for tensorboard
    evaluate_frequency = 10

    if args.use_reward_norm:
        print("------use reward norm------")
        reward_norm = Normalization(shape=args.n_agents)
    elif args.use_reward_scaling:
        print("------use reward scaling------")
        reward_scaling = RewardScaling(shape=args.n_agents, gamma=args.gamma)
    
    total_steps = 0
    i = 0
    repeat_eval = False
    reward_index = 0
    reward_repeat = []
    reward_index_log = 0 
    ep_i = 0

    while 1:
        if ((ep_i%restart_frequency)==0)and(ep_i!=0):
            env.restart()
        obs_n = env.reset_world()
        if args.use_reward_scaling:
            reward_scaling.reset()
        if args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            agent_n.actor.rnn_hidden = None
            agent_n.critic.rnn_hidden = None
        rewards = 0
        succ_list = np.zeros(episode_limit)

        if (i % evaluate_frequency == 0 and i!=0) and not repeat_eval:
            for time_step in range(episode_limit):
                a_n, a_logprob_n = agent_n.choose_action(obs_n, evaluate=True)
                # print(actions.shape)

                obs_next_n, r_n, done_n, succ = env.step_evaluate(a_n)
                # print(a_n)
                succ_list[time_step] = succ
                
                rewards += r_n[0]
                obs_n = obs_next_n
             
                if np.any(done_n):
                    break
                if np.sum(succ_list)==10:
                    break

            if reward_index%3==0 and reward_index!=0:
                repeat_eval = True

            reward_repeat.append(rewards)
                
            if repeat_eval:
                reward_sum_av = np.mean(np.array(reward_repeat))
                reward_sum_std = np.std(np.array(reward_repeat))
                logger.add_scalar('Episode reward', reward_sum_av, reward_index_log)
                logger.add_scalar('Episode reward std', reward_sum_std, reward_index_log)
                reward_index_log+=1
                reward_repeat = []
                
                agent_n.save_model(args.save_dir, args.scenario_name, 0, 222, total_steps)

            reward_index+=1
            
            print("mappo_evaluation_episode:{}".format(reward_index))

        else:
            for t in range(episode_limit):
                a_n, a_logprob_n = agent_n.choose_action(obs_n, evaluate=False)
                s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.

                v_n = agent_n.get_value(s)  # Get the state values (V(s)) of N agents

                # v_n = agent_n.get_value(s)  # Get the state values (V(s)) of N agents
                obs_next_n, r_n, done_n, succ = env.step(a_n)
                succ_list[t] = succ

                if args.use_reward_norm:
                    r_n = reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = reward_scaling(r_n)

                # Store the transition
                replay_buffer.store_transition(t, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

                obs_n = obs_next_n
        
                if np.any(done_n):
                    break
            
            replay_buffer.store_duration(t+1)
            s = np.array(obs_n).flatten()
            v_n = agent_n.get_value(s)
            replay_buffer.store_last_value(t + 1, v_n)

            total_steps += t
            if replay_buffer.episode_num == args.batch_size:
                agent_n.train(replay_buffer, total_steps,logger)  # Training
                replay_buffer.reset_buffer()
            
            i+=1
            repeat_eval = False  
            print("mappo_episode:{}".format(i))
            
        ep_i+=1
        
        if total_steps >= args.max_train_steps:
            break
        # logger.add_scalar('network_loss', loss_sum, i)
            
    logger.close()
    env.shutdown()

def evaluation(args, env, agent_n):
    log_path = args.save_dir+"/log_drone_MPI_V0_MAPPO_evaluation_table"
    logger = SummaryWriter(logdir=log_path) # used for tensorboard
    returns = []
    time_f = []
    s_r = []
    deviation = []
    a_vel = []

    # for agent in agents:
    #     agent.prep_rollouts(device='cpu')

    t_c = 0
    tt = 0
    t_c_ex = 0

    for episode in range(args.evaluate_episodes):
        # reset the environment
        s = env.reset_world()
        rewards = 0
        succ_list = np.zeros(args.episode_limit)
        pos_x = []
        pos_y = []
        
        vel_mag1 = []
        
        for time_step in range(args.episode_limit):
            vel_mag = []
            actions = []
            with torch.no_grad():
                a_n, a_logprob_n = agent_n.choose_action(s, evaluate=True)
            # print(actions.shape)
            s_next, r, done, succ = env.step_evaluate(a_n)
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
    # get the params
    args = get_args()

    set_global_seeds(222)
   
    if args.field_size == 10:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')
    if args.field_size == 15:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_15x15.ttt')
    
    # create multiagent environment
    env = Drone_Env(args,env_name,args.n_agents)
    
    args.high_action = 1
    args.load_buffer = False
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space

    if not args.evaluate:
        args.save_dir = "./" + args.scenario_name + "/model_drone{}".format(args.n_agents)+'/'+'field_size{}'.format(args.field_size)+'/'+ 'env'+strftime("%Y-%m-%d--%H:%M:%S", localtime())
    else:
        args.save_dir = "./" + args.scenario_name + "/model_drone{}".format(args.n_agents)+'/'+'field_size{}'.format(args.field_size)+'/'+ 'env'
    
    args.use_gpu = False
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []        
    for content in env.action_space[:args.n_agents]:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==5) 

    # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
    args.obs_dim = args.obs_shape[0]  # The dimensions of an agent's observation space
    args.action_dim = args.action_shape[0]  # The dimensions of an agent's action space
    args.state_dim = np.sum(args.obs_shape)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
    args.N = args.n_agents

    # Create N agents
    agent_n = MAPPO_MPE(args)
    
    try:
        if args.evaluate:
            evaluation(args,env,agent_n)
        else:
            run(args, env, agent_n)
    except KeyboardInterrupt:
        env.shutdown()

        

