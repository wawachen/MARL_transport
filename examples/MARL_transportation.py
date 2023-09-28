"""
An example of how one might use PyRep to create their RL environments.
In this case, the quadricopter must manipulate a target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
"""
from os.path import dirname, join, abspath
from pyrep import PyRep

from pyrep.envs.Multi_drones_transportation import Drone_Env
from pyrep.policies.buffer import ReplayBuffer
from pyrep.policies.maddpg import MADDPG

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
# from pyrep.robots.end_effectors.uarm_Vacuum_Gripper import UarmVacuumGripper
import matplotlib
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc
import cv2
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

from collections import OrderedDict
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from pyrep.policies.utilities import transpose_list, transpose_to_tensor, hard_update, soft_update
from collections import deque
import progressbar as pb
import math
from pyrep.backend import sim
import imageio


env_name = join(dirname(abspath(__file__)), 'RL_uniform_nobound.ttt')
num_agents = 6 
env = Drone_Env(env_name,num_agents)

action_size = env.action_spec().shape[0]
state_size = env.observation_spec().shape[0]

#parameter settings
## Initialise parameters ##
p = OrderedDict()

## Environment_parameters ##
p.update(num_agents=num_agents, action_size=action_size, action_type = 'continuous', state_size=state_size)

## Episode_parameters ##
p.update(number_of_episodes=1000, episode_length=100, episodes_before_training=300,
                                 learn_steps_per_env_step=3, catchup_tau=.01, catchup_threshold=1.15)
## Replay_Buffer_parameters ##
p.update(buffer_size=50000, n_steps =5)

## Agent_parameters ##
p.update(discount_rate=0.99, tau=0.0001, lr_actor=0.00025, lr_critic=0.0005)

## Model_parameters ##
p.update(batchsize=256, hidden_in_size=300, hidden_out_size=200, l2_decay=0.0001)

## Categorical_parameters ##
p.update(num_atoms=51, vmin=-0.1, vmax=1)

## Noise_parameters ##
p.update(noise_type='BetaNoise', noise_reduction=0.998 , noise_scale_end=0.001,
                               OU_mu=0, OU_theta=0.2, OU_sigma=0.2)

random_seed = np.random.randint(1000)

# set the random seed - this allows for reproducibility
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# how many episodes to save network weights
save_interval = 500
t = 0


# amplitude of noise
# this slowly decreases to 0
noise_reduction = p['noise_reduction'] # each episode we decay the noise by this
noise_scale = np.ones(num_agents) # we start the noise at 1
noise_scale_end = p['noise_scale_end'] # the noise will never drop below this

# some performance metrics to keep track of for graphing afterwards
agent_scores = [[] for _ in range(num_agents)]
agent_scores_last_100 = [deque(maxlen = 100) for _ in range(num_agents)]
agent_scores_avg, previous_agent_scores_avg = np.zeros(num_agents), np.zeros(num_agents)

log_path = os.getcwd()+"/log" # we save tensorboard logs here
model_dir= os.getcwd()+"/model_dir" # we save the model files here, to be reloaded for watching the agents

os.makedirs(model_dir, exist_ok=True) # make the directory if it doesn't exist

# keep 50000 timesteps worth of replay, with n-step-5 bootstraping, and 0.99 discount rate
buffer = ReplayBuffer(size = p['buffer_size'], n_steps = p['n_steps'], discount_rate = p['discount_rate'])

# initialize actor and critic networks and ddpg agents, passing all parameters to it
maddpg = MADDPG(p)

logger = SummaryWriter(logdir=log_path) # used for tensorboard

# training loop
# show progressbar of several metrics of interest
# all the metrics progressbar will keep track of
widget = ['episode: ', pb.Counter(),'/',str(p['number_of_episodes']),' ',
            pb.DynamicMessage('a0_avg_score'), ' ',
            pb.DynamicMessage('a1_avg_score'), ' ',
            pb.DynamicMessage('a0_noise_scale'), ' ',
            pb.DynamicMessage('a1_noise_scale'), ' ',

            pb.DynamicMessage('a2_avg_score'), ' ',
            pb.DynamicMessage('a3_avg_score'), ' ',
            pb.DynamicMessage('a2_noise_scale'), ' ',
            pb.DynamicMessage('a3_noise_scale'), ' ',
            pb.DynamicMessage('a4_avg_score'), ' ',
            pb.DynamicMessage('a5_avg_score'), ' ',
            pb.DynamicMessage('a4_noise_scale'), ' ',
            pb.DynamicMessage('a5_noise_scale'), ' ',
            pb.DynamicMessage('buffer_size'), ' ',
            pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ] 


timer = pb.ProgressBar(widgets=widget, maxval=p['number_of_episodes']).start() # progressbar


for episode in range(0,p['number_of_episodes']):
    training_flag = episode >= p['episodes_before_training'] # the training flag denotes whether to begin training
    if training_flag: # if training
        for agent_num in range(num_agents): # for both agents
            if agent_scores_avg[agent_num]>previous_agent_scores_avg[agent_num]: # if the score is improving
                noise_scale[agent_num] = max(noise_scale[agent_num]*noise_reduction,noise_scale_end) # then reduce noise

    timer.update(episode, a0_avg_score=agent_scores_avg[0], a1_avg_score=agent_scores_avg[1], a2_avg_score=agent_scores_avg[2], a3_avg_score=agent_scores_avg[3],
                a4_avg_score=agent_scores_avg[4], a5_avg_score=agent_scores_avg[5], a0_noise_scale=noise_scale[0], a1_noise_scale=noise_scale[1], a2_noise_scale=noise_scale[2], a3_noise_scale=noise_scale[3], a4_noise_scale=noise_scale[4], a5_noise_scale=noise_scale[5], buffer_size=len(buffer)) # progressbar

    buffer.reset() # reseting the buffer is neccessary to ensure the n-step bootstraps reset after each episode
    reward_this_episode = np.zeros(num_agents) # keeping track of episode reward
    # if training_flag: # useful for viewing the agent in full screen, otherwise leave as is
    #     env_info = env.reset(train_mode=True)[brain_name]
    # else:
    #     env_info = env.reset(train_mode=True)[brain_name]
    env._reset()   
    states = env._get_state() # we get states directly from the environment
    obs = [states[i,:] for i in range(num_agents)] # and reshape them into a list

    # save model or not this episode
    save_info = ((episode) % save_interval < 1 or episode==p['number_of_episodes']-1)
    t_d = True
    
    for episode_t in range(p['episode_length']):
        env.agents[0].agent.get_panoramic()
        t += 1 # increment timestep counter
        # explore only for a certain number of episodes
        # action input needs to be transposed
        # actions = maddpg.act(transpose_to_tensor([obs]), noise_scale=noise_scale) # the actors actions
        # actions_array = torch.stack(actions).detach().numpy().squeeze() # converted into a np.array
        # print(obs)
        if training_flag:
            #action is the angle and speed
            actions = maddpg.act(transpose_to_tensor([obs]), noise_scale=noise_scale) # the actors actions
            actions_array = torch.stack(actions).detach().numpy().squeeze() # converted into a np.array
            random_bearings1 = actions_array[:,0]*180
            random_bearings = random_bearings1/180 * np.pi
            speeds = (actions_array[:,1]+1)/2
            velocity_x = speeds * np.r_[[math.cos(random_bearings[i]) for i in range(6)]]
            velocity_y = speeds * np.r_[[math.sin(random_bearings[i]) for i in range(6)]]
            actions_array_cmd = np.r_[[(velocity_x[j],velocity_y[j]) for j in range(6)]] # behave randomly before training

            # print(actions_array)
        else:
            if t_d:
                random_bearings1 = np.random.uniform(-180, 180, 6)
            else:
                random_list = [np.random.uniform(random_b_old[ii]-30,random_b_old[ii]+30) for ii in range(6)]
                random_bearings1 = np.r_[random_list]
            
            #convert into radian
            random_bearings = random_bearings1/180 * np.pi
            #generate velocity cmd
            speeds = np.random.uniform(0, 1, 6)
            actions_array = np.r_[[(random_bearings1[j]/180,speeds[j]*2-1) for j in range(6)]]

            velocity_x = speeds * np.r_[[math.cos(random_bearings[i]) for i in range(6)]]
            velocity_y = speeds * np.r_[[math.sin(random_bearings[i]) for i in range(6)]]
            actions_array_cmd = np.r_[[(velocity_x[j],velocity_y[j]) for j in range(6)]] # behave randomly before training
            random_b_old = random_bearings1
            t_d = False

            
        #print(actions_array)
        env_info = env._step(actions_array_cmd)   # input the actions into the env

        next_states = env_info[0] # get the next states
        next_obs = [next_states[i,:] for i in range(num_agents)] # and reshape them into a list 

        rewards = np.array(env_info[1]) # get the rewards
        dones = np.array(env_info[2]) # and whether the env is done

        # add data to buffer
        transition = ([obs, actions_array, rewards, next_obs, dones])
        buffer.push(transition)

        obs = next_obs # after each timestep update the obs to the new obs before restarting the loop
        
        # for calculating rewards for this particular episode - addition of all time steps
        reward_this_episode += rewards
        previous_agent_scores_avg = agent_scores_avg
        
        if np.any(dones):                                  # exit loop if episode finished
            break
            
    # update the episode scores being kept track of - episode score, last 100 scores, and rolling average scores
    for agent_num in range(num_agents):
        previous_agent_scores_avg[agent_num] = agent_scores_avg[agent_num]
        agent_scores[agent_num].append(reward_this_episode[agent_num])
        agent_scores_last_100[agent_num].append(reward_this_episode[agent_num])
        agent_scores_avg[agent_num] = np.mean(agent_scores_last_100[agent_num])

    # update agents networks
    if (len(buffer) > p['batchsize']) & training_flag:
        for _ in range(p['learn_steps_per_env_step']): # learn multiple times at every step
            for agent_num in range(p['num_agents']): # for both agents
                # if agent_scores_avg[agent_num] < (p['catchup_threshold']*min(agent_scores_avg)+0.01):# if agent too far ahead then wait
                samples = buffer.sample(p['batchsize']) # sample the buffer
                #print(samples)
                maddpg.update(samples, agent_num, logger=logger) # update the agent
                maddpg.update_targets(agent_num) # soft update the target network towards the actual networks
                #if t % C == 0:
                #    maddpg.hard_update_targets(agent_num) # hard update the target network towards the actual networks
                # this can be used instead of soft updates
                
                # else:
                #     # update the target networks of the worse agent towards the better one
                #     soft_update(maddpg.maddpg_agent[1-agent_num].actor,maddpg.maddpg_agent[agent_num].actor,p['catchup_tau'])
                #     soft_update(maddpg.maddpg_agent[1-agent_num].critic,maddpg.maddpg_agent[agent_num].critic,p['catchup_tau'])
    
    # add average score to tensorboard
    if (episode % 100 == 0) or (episode == p['number_of_episodes']-1):
        for a_i, avg_rew in enumerate(agent_scores_avg):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)
            
    '''
    #saving model
    save_dict_list =[]
    if save_info:
        for i in range(num_agents):
            save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                         'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                         'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                         'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
            save_dict_list.append(save_dict)
            torch.save(save_dict_list, os.path.join(model_dir, 'episode-{}.pt'.format(episode)))
    '''
env.shutdown()
logger.close()
timer.finish()

