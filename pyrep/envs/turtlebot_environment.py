from os import path
from pyrep import PyRep
from pyrep.envs.turtle_RL_agent import Turtle_o

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

import random
from gym import spaces
import numpy as np
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Turtle_Env:

    def __init__(self,env_name,num_agents):
        self.reset_callback = self.reset_world
        self.reward_callback = self.reward_and_teriminate
        self.observation_callback = self.observation
        # self.done_callback = self.done

        # environment parameters
        self.discrete_action_space = False
        self.time_step = 0

        self.env_name = env_name
        self.close_simulation = False

        # configure spaces
        self.x_limit_min = -5+0.2
        self.x_limit_max = 5-0.2
        self.y_limit_min = -5+0.2
        self.y_limit_max = 5-0.2
        self.safe_distance = 0.3545

        self.action_space = []
        self.observation_space = []
        self.num_a = num_agents
        
        self.shared_reward = True
        self.sight_range = 2.0
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
       
        self.model_handles,ii = self.import_agent_models()
       
        self.agents = [Turtle_o(i) for i in range(num_agents)] #because it only import odd index 
        self.goals = self.generate_goal()

        np.random.seed(125)

        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(5) #3*3
            else:
                u_action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

            total_action_space.append(u_action_space)
            self.action_space.append(total_action_space[0])
            #observation space
            obs_dim = len(self.observation_callback(agent))
            #print(obs_dim)
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            # agent.action.c = np.zeros(self.world.dim_c)

    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        #pr.remove_model(m1)
        model_handles = []
        objs = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'turtlebot_beta.ttm'))
            model_handles.append(m1)
            objs.append(m)

        return model_handles,objs

    # def is_collision(self, m):
    #     l = []
    #     for wall in self.walls:
    #         if self.base[m].check_collision(wall):
    #             l.append(1)
    #     for i,base in enumerate(self.base):
    #         if i==m: continue
    #         if self.base[m].check_collision(base):
    #             l.append(1)
    #     return np.any(l)


    def generate_goal(self):
        #visualization goal
        targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.36, 0.36, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_a)]

        goal_points = np.array([[-2.8,0,0.005],[0,0,0.005],[2.8,0,0.005]]) #[-1.25,0,0.005],[0,1.25,0.005],[1.25,0,0.005],[1.25,1.25,0.005],[-1.25,-1.25,0.005],[0,-1.25,0.005]

        for i in range(self.num_a):
            targets[i].set_position(goal_points[i])

        return goal_points

    def check_collision(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance < 1.0:
            return 1
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.agent.get_position()[:2] - agent2.agent.get_position()[:2]
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return True if dist < self.safe_distance else False

    # def random_spread_without_obstacle(self):
    #     self.agents = []
    #     model_handles = []
    #     objs = []
    #     robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
    #     for i in range(self.num_a):
    #         [m,m1]= self.pr.import_model(path.join(robot_DIR, 'turtlebot_beta.ttm'))
    #         model_handles.append(m1)
    #         objs.append(m)
    #         self.agents.append(Turtle(i))
            
    #         vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
    #         self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],0.0607,0.0,0.0,np.radians(random.uniform(-180,180))])
    #         check_conditions = self.agents[i].agent.assess_collision()

    #         while check_conditions:
    #             vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
    #             self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],0.0607,0.0,0.0,np.radians(random.uniform(-180,180))])
    #             check_conditions = self.agents[i].agent.assess_collision()

    #         # print("all",vpts)
    #         # print("current",vpt)
            
    #     return model_handles,objs

    def random_position_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        saved_agents = []
        vpts = []
        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'turtlebot_beta.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Turtle_o(i))

            self.agents[i].agent.set_motor_locked_at_zero_velocity(True)
            if i == 0:
                self.agents[i].agent.set_3d_pose([random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max), 0.0607,0.0,0.0,np.radians(random.uniform(-180,180))])
                vx = self.agents[i].agent.get_position()[0]
                vy = self.agents[i].agent.get_position()[1]
                vpts.append([vx,vy])
                saved_agents.append(i)
            else:
                vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
                check_conditions = np.sum(check_list)

                while check_conditions:
                    vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                    check_list = [self.check_collision(vpt,vpts[m]) for m in saved_agents]
                    check_conditions = np.sum(check_list)

                # print("all",vpts)
                # print("current",vpt)
                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],0.0607,0.0,0.0,np.radians(random.uniform(-180,180))])
                vpts.append(vpt)
                saved_agents.append(i)
        print(vpts)
        return model_handles,objs

    def direct_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"

        if self.num_a == 3:
            pos = np.array([[2.0,2.5],[-2.5,2.5],[0,-2.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)
        
        if self.num_a == 4:
            pos = np.array([[3.5,4.5],[3.5,-4.5],[-3.5,-4.5],[-3.5,4.5]])
            # pos = np.array([[2.5,4.5],[3.5,4.5],[-2.5,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        if self.num_a == 6:
            pos = np.array([[4.5,4.5],[4.5,-4.5],[-4.5,4.5],[-4.5,-4.5],[0,4.5],[0,-4.5]])
            # pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        for i in range(self.num_a):
            # [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            # model_handles.append(m1)
            # objs.append(m)
            # self.agents.append(Drone_s(i))
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'turtlebot_hybrid.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Turtle_o(i))

            self.agents[i].agent.set_motor_locked_at_zero_velocity(True)
            #0.0607
            self.agents[i].agent.set_3d_pose([pos[i,0],pos[i,1],0.0607,0.0,0.0,0.0])
                

        return model_handles,objs


    def reset_world(self):
        #self.suction_cup.release()
        if not self.close_simulation:
            for i in range(self.num_a):
                self.pr.remove_model(self.model_handles[i])
      
        # self.model_handles,ii = self.import_agent_models()
        # start_index = int(ii[0].get_name().split("#")[1])
        # print(start_index)
        # self.agents = [Turtle(i) for i in range(self.num_a)]
        self.model_handles,ii = self.random_position_spread()
        if self.close_simulation:
            self.goals = self.generate_goal()

        self.time_step += 1

        obs_n = []
        for agent in self.agents:
            obs_n.append(self.observation_callback(agent))

        self.close_simulation = False

        return obs_n
    
    def restart(self):
        if self.pr.running:
            self.pr.stop()
        self.pr.shutdown()

        self.pr = PyRep()
        self.pr.launch(self.env_name, headless=False)
        self.pr.start()
        self.close_simulation = True

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        for j in range(10):
            # set action for each agent
            for i, agent in enumerate(self.agents):
                agent.set_action(action_n[i])
            self.pr.step()

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(self.observation_callback(agent))
            rw,ter = self.reward_callback(agent,i)
            reward_n.append(rw)
            done_n.append(ter)

        #all agents get total reward in cooperative case
        reward = np.sum(reward_n[:]) #need modify
        if self.shared_reward:
            reward_n = [reward] * self.num_a

        # print(obs_n[0].shape)

        return np.array(obs_n), np.array(reward_n), np.array(done_n), info_n

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def reward_and_teriminate(self, agent, m):
       # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        terminate = 0
        finish_sig = np.zeros(self.num_a)

        for i in range(self.goals.shape[0]):
            dists = [np.sqrt(np.sum(np.square(a.agent.get_position()[:2] - self.goals[i,:2]))) for a in self.agents]
            finish_sig[i] = np.any((np.array(dists)<0.3))
            rew -= min(dists)
        
        if np.all(finish_sig):
            terminate = 1
            rew = 0
            rew += 1

        # collision detection
        wall_dists = np.array([np.abs(5-agent.agent.get_position()[1]),np.abs(5+agent.agent.get_position()[1]),np.abs(5+agent.agent.get_position()[0]),np.abs(5-agent.agent.get_position()[0])]) # rangefinder: forward, back, left, right
        wall_sig = np.any(wall_dists<0.206)

        agent_collision = []
        for a in self.agents:
            if a == agent: continue
            if self.is_collision(agent,a):
                agent_collision.append(1)
            else:
                agent_collision.append(0)
        agent_sig = np.any(np.array(agent_collision))

        # if agent.agent.assess_collision():
        #     rew-=1
        #     terminate = 1
        if wall_sig or agent_sig:
            rew-=1
            terminate = 1

        # for agent in self.agents:
        #     print(agent.agent.assess_collision())
        #     if agent.agent.assess_collision():
        #         rew-=1
        #         terminate = 1
        #         break

        # if self.is_collision(m):
        #     rew -= 1
        
        return rew,terminate

    def get_distance(self,p1,p2):
        return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def get_bearing(self,p1,p2):
        ang = math.atan2(p2[0]-p1[0], p2[1]-p1[1])

        if ang<=-np.pi/2:
           return ang + 3*np.pi/2
        else:
           return ang - np.pi/2
        
    def get_turning_angle(self,a,b):
        a1 = np.sqrt((a[0] * a[0]) + (a[1] * a[1]))
        b1 = np.sqrt((b[0] * b[0]) + (b[1] * b[1]))
        aXb = (a[0] * b[0]) + (a[1] * b[1])

        cos_ab = aXb/(a1*b1)
        angle_ab = math.acos(cos_ab)*(180.0/np.pi)
        return angle_ab

    def point_direction(self,a,b,c):
        #start, end , point S = (x1-x3)*(y2-y3)-(y1-y3)*(x2-x3) 
        #S>0, left; S<0, right; S=0, on the line
        S = (a[0]-c[0])*(b[1]-c[1])-(a[1]-c[1])*(b[0]-c[0])
        if S < 0:
            return 1
        if S > 0:
            return -1
        if S == 0:
            return 1

    def get_local_goal(self,agent,i):
        pos = agent.agent.get_position()[:2]
        orientation = agent.agent.get_orientation()[2]
        x = pos[0]
        y = pos[1]
        theta = orientation

        goal_x = self.goals[i][0]
        goal_y = self.goals[i][1]

        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def observation(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i in range(self.goals.shape[0]):  # world.entities:
            entity_pos.append(np.array(self.get_local_goal(agent,i))/10.0)   
        # entity_pos.append(agent.target-agent.state.p_pos)
        four_sides =  []
        for reading in agent.agent.get_proximity():
            if reading == -1:
                four_sides.append(0)
            else:
                four_sides.append(1)
        # print(four_sides)
        
        # communication of all other agents
        comm = []
        other_pos = []
        for other in self.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append((other.agent.get_position(agent.agent)[:2])/10.0)

        # pos_obs = agent.agent.get_position()[:2]/2.5

        return np.concatenate([np.array(agent.agent.get_base_velocities())] + entity_pos + other_pos)

    # def observation(self, agent):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for i in range(self.goals.shape[0]):  # world.entities:
    #         # distance = np.sqrt(np.sum(np.square(self.goals[i,:2] - agent.agent.get_position()[:2])))
    #         # if distance>self.sight_range:
    #         #     entity_pos.append([0,0])
    #         # else:
    #         entity_pos.append(np.array(self.get_local_goal(agent,i))/5.0)   
    #     # entity_pos.append(agent.target-agent.state.p_pos)
    #     # four_sides =  []
    #     # for reading in agent.agent.get_proximity():
    #     #     if reading == -1:
    #     #         four_sides.append(0)
    #     #     else:
    #     #         four_sides.append(1)
    #     # print(four_sides)
        
    #     # communication of all other agents
    #     comm = []
    #     other_pos = []
    #     for other in self.agents:
    #         if other is agent: continue
    #         distance = np.sqrt(np.sum(np.square(other.agent.get_position()[:2] - agent.agent.get_position()[:2])))
    #         # comm.append(other.state.c)
    #         if distance > self.sight_range:
    #             other_pos.append([0,0])
    #         else:
    #             other_pos.append((other.agent.get_position(agent.agent)[:2])/5.0)

    #     # pos_obs = agent.agent.get_position()[:2]/2.5

    #     return np.concatenate([np.array(agent.agent.get_base_velocities())] + entity_pos + other_pos)

    




