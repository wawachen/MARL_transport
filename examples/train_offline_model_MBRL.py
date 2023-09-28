import torch
import numpy as np
from pyrep.policies.MPC import MPC 
from pyrep.envs.Agent import Agent
from tqdm import trange
from scipy.io import savemat
from tensorboardX import SummaryWriter
from time import localtime, strftime
import os
from pyrep.envs.orca_bacterium_environment_demo_v0_MBRL import Drone_Env
from pyrep.common.arguments_v0 import get_args
from os.path import dirname, join, abspath
from dotmap import DotMap
import math

if __name__=="__main__":
    n_agent = 4
    training = 1

    torch.manual_seed(200)
    # torch.cuda.manual_seed_all(200)
    np.random.seed(200)

    # get the params
    args = get_args()
    num_agents = n_agent
    args.field_size = 10
    args.load_type = "four"
    args.n_agents = n_agent

    env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')
   
    env = Drone_Env(args, env_name,num_agents)

    # For MBRL
    if n_agent == 3:
        task_hor = 50
    if n_agent == 4:
        task_hor = 60
    if n_agent == 6:
        task_hor = 70

    #For mpc
    params = DotMap()
    params.per = 1
    params.prop_mode = "TSinf"
    params.opt_mode = "CEM"
    params.npart = 20
    params.ign_var = False
    params.plan_hor = 5 
    params.num_nets = 1
    params.epsilon = 0.001
    params.alpha = 0.25
    params.epochs = 25 #5
    params.model_3d_in = 4*n_agent+2*n_agent
    params.model_3d_out = 4*n_agent 
    params.popsize = 50 
    params.max_iters = 3
    params.num_elites = 10
    params.load_model = (not training)

    policy = MPC(params,env)
    
    agent = Agent(env)
    ntrain_iters = 50
    nrollouts_per_iter = 5
    ninit_rollouts = 5
    neval = 1

    if training:
        log_path = os.path.join("/home/wawa/RL_transport_3D/log_MBRL_model",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
    else:
        log_path = os.path.join("/home/wawa/RL_transport_3D/log_MBRL_model")

    os.makedirs(log_path, exist_ok=True)
    logger = SummaryWriter(logdir=log_path) # used for tensorboard

    if training:

        demoDataFile = 'orca_demonstrationMBRL_ep100_{0}agents_env10.npz'.format(n_agent)
        demoData = np.load(demoDataFile)
        
        """episode_batch:  self.demo_episode x timestep x keydim   
        """
        action_episodes = demoData['acs']
        obs_episodes = demoData['obs']
        obs_next_episodes = demoData['obs_next']

        assert(action_episodes.shape[0]==100)
        # Perform initial rollouts
        samples = []

        train_obs = obs_episodes.reshape(-1, obs_episodes.shape[-1]) 
        train_acs = action_episodes.reshape(-1, action_episodes.shape[-1])
        train_obs_next = obs_next_episodes.reshape(-1, obs_next_episodes.shape[-1])

        #samples [episode, steps,n]
        policy.train(train_obs, train_acs,train_obs_next, logger)

        # Training loop
        for i in trange(ntrain_iters):
            print("####################################################################")
            print("Starting training iteration %d." % (i + 1))

            samples = []

            #horizon, policy, wind_test_type, adapt_size=None, log_data=None, data_path=None
            #MBRL is baseline no need to log data in agent sampling
            for j in range(max(neval, nrollouts_per_iter)):
                samples.append(
                        agent.sample(task_hor, policy)
                    )
            # print("Rewards obtained:", [sample["reward_sum"] for sample in samples[:self.neval]])
            logger.add_scalar('Reward', np.mean([sample["reward_sum"] for sample in samples[:]]), i)
            samples = samples[:nrollouts_per_iter]

            if i < ntrain_iters - 1:
                #add new samples into the whole dataset and train the whole dataset
                policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["obs_next"] for sample in samples],
                    logger
                )

        logger.close()
    else:
        ###########################################################################
        returns = []
        time_f = []
        s_r = []
        deviation = []
        a_vel = []

        t_c = 0
        tt = 0
        t_c_ex = 0

        for episode in range(20):
            # reset the environment
            s = env.reset_world()
            rewards = 0
            succ_list = np.zeros(args.evaluate_episode_len)
            pos_x = []
            pos_y = []
            
            vel_mag1 = []
            policy.reset()
            
            for time_step in range(task_hor):
                vel_mag = []
                actions = []
                with torch.no_grad():
                    action,act_l,store_top_s,store_bad_s = policy.act(s, time_step, env.goals[:,:2]) #[6,5,2] store top s
                # print(actions.shape)
                s_next, reward, done, succ = env.step_evaluate(action)
                succ_list[time_step] = succ
                for j in range(args.n_agents):
                    pos_x.append(s_next[j*2]*env.field_size)
                    pos_y.append(s_next[j*2+1]*env.field_size)    
                    vel_mag.append(math.sqrt(s_next[args.n_agents*2+j*2]**2+s_next[args.n_agents*2+j*2+1]**2))

                vel_mag1.append(np.sum(np.array(vel_mag))/len(vel_mag))  
                
                rewards += reward[0]
                s = s_next
                t_c += 1
                if np.any(done):
                    break
                if np.sum(succ_list)==10:
                    break
                
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
            print("time_steps",time_step)
        print("Results is")
        print("Finished time: ", np.sum(np.array(time_f))/len(time_f), ', ', np.std(np.array(time_f)) )
        print('Average speed: ', np.sum(np.array(a_vel))/len(a_vel), ', ', np.std(np.array(a_vel)))
        print("Success rate: ", np.sum(np.array(s_r))/len(s_r))
        print("Deviation: ", np.sum(np.array(deviation))/len(deviation),', ', np.std(np.array(deviation)))
        print("Total rewards", np.sum(np.array(returns)))
        logger.close()




