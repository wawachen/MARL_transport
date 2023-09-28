#!/usr/bin/env python 
from os.path import dirname, join, abspath

import os
from pyrep.envs.bacterium_environment_mpi_pc_global_5m_wt_ddpg import Drone_Env
from pyrep.common.arguments_v0 import get_args
from pyrep.policies.maddpg_drone_att_pc1_shared_wt_ddpg import MADDPG
# from pyrep.policies.maddpg_drone_scratch import MADDPG
from pyrep.common.replay_buffer_new import Buffer
import numpy as np
from tensorboardX import SummaryWriter
import torch
from time import localtime, strftime
import random
import math
from scipy.io import savemat

def run(args, env, agent):
    noise = args.noise_rate
    epsilon = args.epsilon
    episode_limit = args.max_episode_len
    max_episodes = args.max_episodes
    restart_frequency = 1000
    buffer = Buffer(args)

    log_path = args.save_dir+"/log_drone_curriculum"
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    evaluate_frequency = 10
    i = 0
    repeat_eval = False
    reward_index = 0
    reward_repeat = []
    reward_index_log = 0 
    ep_i = 0

    while 1:
        if ((ep_i%restart_frequency)==0)and(ep_i!=0):
            env.restart()
    
        s = env.reset_world()
        rewards = 0
        succ_list = np.zeros(episode_limit)
        
        agent.prep_rollouts(device='cpu')

        if (i % evaluate_frequency == 0 and i!=0) and not repeat_eval:
            for time_step in range(episode_limit):
                with torch.no_grad():
                    actions = agent.select_actions(s, 0.05, 0.05, args.use_gpu, logger)
                # print(actions.shape)
                s_next, r, done, succ = env.step_evaluate(actions)
                succ_list[time_step] = succ
                
                rewards += np.sum(np.array(r))
                s = s_next
             
                if np.any(done):
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

            reward_index+=1
            print("curriculum_evaluation_episode:{}".format(reward_index))

        else:
            for t in range(episode_limit):
                ###########
                with torch.no_grad():                    
                    actions = agent.select_actions(s, noise, epsilon, args.use_gpu, logger)
                
                ###########
                s_next, r, done = env.step(actions)

                ###########
                buffer.store_episode(s[:args.n_agents], actions, r[:args.n_agents], s_next[:args.n_agents],done[:args.n_agents])

                s = s_next
                
                if buffer.current_size >= args.batch_size:
                    agent.prep_training(device='cpu')
                    transitions = buffer.sample(args.batch_size)
                
                    agent.train(transitions, logger,args.use_gpu,i)
                    agent.prep_rollouts(device='cpu')

                noise = max(0.05, noise - 0.0000005)
                epsilon = max(0.05, noise - 0.0000005)

                if np.any(done):
                    break
    
            i+=1
            repeat_eval = False
               
            print("curriculum_episode:{}".format(i))

        ep_i+=1
        
        if i==max_episodes:
            break
            
    logger.close()
    env.shutdown()

def generate_evaluation_curve(args,env,agents):
    log_path = args.save_dir+"/log_drone_MPI_V0_wt_ddpg_evaluation" 
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
        print("demo_evaluation_episode{0}: {1}".format(i,np.mean(returns)))

    mdict = {'mean':reward_all,'std':std_all}
    savemat(args.save_dir,mdict)


def evaluation(args, env, agent):
    log_path = args.save_dir+"/log_drone_MPI_shared_evaluation_wt_route1"
    logger = SummaryWriter(logdir=log_path) # used for tensorboard
    returns = []
    time_f = []
    s_r = []
    deviation = []
    a_vel = []
        
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

        goal_t = env.goals
        agent.prep_rollouts(device='cpu')
        
        for time_step in range(args.evaluate_episode_len):
            vel_mag = []
            actions = []
            with torch.no_grad():
                actions = agent.select_actions(s, 0.05, 0.05, args.use_gpu, logger)
            
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
            
            goals = goal_t
            
            assert(args.n_agents==goals.shape[0])
            
            dev_all = []
            for m in range(10):
                d_e_t =  0
                for i in range(args.n_agents):
                    d_e = np.zeros(args.n_agents)
                    for j in range(args.n_agents):
                        d_e[j] = np.sqrt(np.sum((goals[i,:2]-np.array([pos_x_end[args.n_agents*m:args.n_agents*m+args.n_agents][j],pos_y_end[args.n_agents*m:args.n_agents*m+args.n_agents][j]]))**2))
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
    if args.field_size == 5:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_square_six_ss.ttt')
    if args.field_size == 6:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_square_six_ss1.ttt')
        

    # create multiagent environment
    env = Drone_Env(args, env_name)
    
    args.high_action = 1
    args.load_buffer = False
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # observation space
    args.save_dir = "./" + args.scenario_name + "/stage{}".format(args.stage) +"/model_drone{}".format(args.n_agents)

    if not args.evaluate:
        args.save_dir = "./" + args.scenario_name + "/stage{}".format(args.stage) +"/model_drone{}".format(args.n_agents)+strftime("%Y-%m-%d--%H:%M:%S", localtime())
    else:
        args.save_dir = "./" + args.scenario_name + "/stage{}".format(args.stage) +"/model_drone{}".format(args.n_agents)
    
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

    agent = MADDPG(args) 

    try:
        if args.evaluate:
            evaluation(args, env, agent)
        else:
            run(args, env, agent)
    except KeyboardInterrupt:
        env.shutdown()

