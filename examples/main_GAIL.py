#!/usr/bin/env python
# Created at 2020/3/14

import torch
import numpy as np
from time import localtime, strftime
import random
from os.path import dirname, join, abspath
from pyrep.envs.bacterium_environment_mpi_RL_gail import Drone_Env
from tensorboardX import SummaryWriter
import math
from pyrep.common.arguments_gail1 import get_args
from pyrep.baselines.replay_memory import Memory
import time
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyrep.baselines.torch_u import *
from pyrep.baselines.gail_nets import GAIL

device = "cpu"

def to_device(device, *args):
    return [x.to(device) for x in args]

def run(args, env, agent, expert_traj):
    episode_limit = args.max_episode_len
    max_episodes = args.max_episodes
    restart_frequency = 1000
    evaluate_frequency = 10

    i = 0
    repeat_eval = False
    reward_index = 0
    reward_repeat = []
    reward_index_log = 0 
    ep_steps = 0
    
    while 1:
        rewards = 0
        succ_list = np.zeros(episode_limit)

        if (i % evaluate_frequency == 0 and i!=0) and not repeat_eval:
            if ((ep_steps%restart_frequency)==0)and(ep_steps!=0):
                env.restart()
            s = env.reset_world()
            for time_step in range(episode_limit):
                state_var = tensor(s).unsqueeze(0)
                with torch.no_grad():
                    action = agent.select_action_eval(state_var)
                    action = action.astype(np.float64)
            
                s_next, r, done, succ = env.step(action)

                succ_list[time_step] = succ
                rewards += r[0]
                s = s_next

                if np.sum(succ_list)==10:
                    break
             
                if done[0]:
                    break

            if reward_index%3==0 and reward_index!=0:
                repeat_eval = True

            reward_index+=1
            reward_repeat.append(rewards)

            if repeat_eval:
                reward_sum_av = np.mean(np.array(reward_repeat))
                reward_sum_std = np.std(np.array(reward_repeat))
                logger.add_scalar('Episode reward', reward_sum_av, reward_index_log)
                logger.add_scalar('Episode reward std', reward_sum_std, reward_index_log)
                logger.add_scalar('Total episode reward', reward_sum_av, ep_steps)
                logger.add_scalar('Total episode reward std', reward_sum_std, ep_steps)
                reward_index_log+=1
                reward_repeat = []

            print("gail_evaluation_episode:{}".format(reward_index))
            ep_steps+=1
        else:
            # discrim_net.to(torch.device('cpu'))
            memory = Memory()

            num_steps = 0

            while num_steps < args.min_batch_size:
                if ((ep_steps%restart_frequency)==0)and(ep_steps!=0):
                    env.restart()
                state = env.reset_world()

                for t in range(10000):
                    state_var = tensor(state).unsqueeze(0)

                    print("Episode{0}-Collect sampling data now: {1}".format(i,len(memory)))
                    
                    with torch.no_grad():
                        action = agent.select_action(state_var)
                        action = action.astype(np.float64)

                    next_state, reward, done, _ = env.step(action)

                    state_action = tensor(np.hstack([state, action]), dtype=dtype)

                    with torch.no_grad():
                        reward = -math.log(agent.discrim_net(state_action)[0].item())

                    mask = 0 if np.any(done) else 1

                    memory.push(state, action, mask, next_state, reward)

                    state = next_state
                    
                    if np.any(done):
                        break

                # log stats
                num_steps += (t + 1)
                ep_steps +=1

            batch = memory.sample()

            agent.update_params(batch, expert_traj, i, logger)
            
            i+=1
            repeat_eval = False
            
            print("gail_episode:{}".format(i))

            if args.save_model_interval > 0 and (i+1) % args.save_model_interval == 0:       
                agent.save_model(i)

        if i==max_episodes:
            break
        # logger.add_scalar('network_loss', loss_sum, i)
            
    logger.close()
    env.shutdown()


def evaluation(args, env, agents):
    log_path = args.save_dir+"/log_drone_MPI_gail_evaluation_table"
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    print(f"Loading Pre-trained MAGAIL model from {log_path}!!!")
    agents.load_model(args.save_dir)

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
        
        for time_step in range(args.evaluate_episode_len):
            vel_mag = []
            actions = []
            with torch.no_grad():
                actions = agents.get_action_log_prob(s if len(s.shape) > 1 else s.unsqueeze(-1))
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
    args = get_args()
    # set_global_seeds(222)
   
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
        action_shape.append(content.shape[0])
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==2) 

    log_path = args.save_dir+"/log_drone_MPI_V0" 
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    args.general_expert_data_path = 'orca_demonstration_ep3000_{}agents_env{}.npz'.format(args.n_agents,args.field_size)

    dtype = torch.float64
    torch.set_default_dtype(dtype)

    state_dim = env.observation_space[0].shape[0]*args.n_agents
    is_disc_action = 0
    action_dim = action_shape[0]*args.n_agents

    assert state_dim == 18*args.n_agents
    assert action_dim == 2*args.n_agents

    """define actor and critic"""
    agent = GAIL(args, state_dim, action_dim)
    
    # load trajectory
    demoData = np.load(args.general_expert_data_path)
    
    """episode_batch:  self.demo_episode x timestep x n x keydim   
    """
    action_episodes = demoData['acs']
    obs_episodes = demoData['obs']
    obs_next_episodes = demoData['obs_next']

    assert(action_episodes.shape[0]==3000)
    assert(action_episodes.shape[1]==25)

    if args.n_agents == 3:
        demo_timesteps = 14
    if args.n_agents == 4:
        demo_timesteps = 25
    if args.n_agents == 6:
        demo_timesteps = 35

    expert_state = []
    expert_action = []
    expert_next_state = []
    for i in range(action_episodes.shape[0]):
        for j in range(demo_timesteps):
            state_temp = []
            action_temp = []
            next_state_temp = []
            for m in range(args.n_agents):
                state_temp.append(obs_episodes[i,j,m,:])
                action_temp.append(action_episodes[i,j,m,:])
                next_state_temp.append(obs_next_episodes[i,j,m,:])
            # print(state_temp)
            expert_state.append(np.concatenate(state_temp,axis=0))
            expert_action.append(np.concatenate(action_temp,axis=0))
            expert_next_state.append(np.concatenate(next_state_temp,axis=0))
    
    expert_state = np.array(expert_state)
    expert_action = np.array(expert_action)
    expert_next_state = np.array(expert_next_state)
    expert_traj = np.concatenate([expert_state,expert_action],axis = 1)

    assert expert_traj.shape[0]==3000*14

    try:
        if args.evaluate:
            # evaluation(args,env,agents)
            pass
        else:
            run(args, env, agent, expert_traj)

    except KeyboardInterrupt:
        env.shutdown()
