from os import path
from pyrep import PyRep
from pyrep.envs.drone_RL_agent import Drone_s
from pyrep.envs.drone_RL_agent import Drone_s_w

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
import random

import gym
from gym import spaces
import numpy as np
import math
import random
from scipy.optimize import linear_sum_assignment


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Drone_Env:

    def __init__(self,args,env_name,num_agents):
        ######################################################
        # random.seed(seed)
        # np.random.seed(seed)
        ######################################################
        self.args = args
        self.reset_callback = self.reset_world
        self.reward_callback = self.reward_and_terminate

        self.is_pc = self.args.is_pc
        self.is_local_obs = self.args.is_local_obs
        self.is_sensor_obs = self.args.is_sensor_obs

        if self.is_local_obs:
            print("use local sight")
            self.sight = args.local_sight
            self.observation_callback = self.local_observation
        elif self.is_sensor_obs:
            print("use sensor obs")
            self.observation_callback = self.sensor_observation
        else:
            self.observation_callback = self.observation

        # self.done_callback = self.done

        # environment parameters
        self.discrete_action_space = False
        self.time_step = 0
        self.field_size = self.args.field_size/2 #field_size can 10 or 15

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.num_a = num_agents
        self.env_name = env_name
        self.close_simulation = False

        self.safe_distance = 0.5
        self.x_limit_min = -self.field_size+self.safe_distance/2+0.5
        self.x_limit_max = self.field_size-self.safe_distance/2-0.5
        self.y_limit_min = -self.field_size+self.safe_distance/2+0.5
        self.y_limit_max = self.field_size-self.safe_distance/2-0.5

        self.shared_reward = True
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles, self.load_handle = self.import_agent_models()
        if self.is_sensor_obs:
            self.agents = [Drone_s_w(i) for i in range(num_agents)]
        else:
            self.agents = [Drone_s(i) for i in range(num_agents)]

        self.payload = Shape('Cuboid4')
        self.targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.56, 0.56, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_a)]

        self.goals = self.generate_goal()

        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(5) #3*3
            else:
                u_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

            total_action_space.append(u_action_space)
            self.action_space.append(total_action_space[0])
            #observation space
            obs_dim = len(self.observation_callback(agent))
            # print(obs_dim)
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"   
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            if self.is_sensor_obs:
                [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL.ttm'))
                model_handles.append(m1)
            else:
                [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
                model_handles.append(m1)

        if self.args.load_type == "three":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_three.ttm'))
        if self.args.load_type == "four":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_four.ttm'))
        if self.args.load_type == "six":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_six.ttm'))

        return model_handles,m2

    
    def check_collision_a(self,agent1,agent2):
        delta_pos = agent1.agent.get_drone_position()[:2] - agent2.agent.get_drone_position()[:2]
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        return True if dist <= self.safe_distance else False

    def check_collision_p(self,point,point1):
        distance = math.sqrt((point[0]-point1[0])**2+(point[1]-point1[1])**2) 
        if distance < 2.5:
            return 1
        else:
            return 0

    def load_spread(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"  

        ################
        if self.args.load_type == "three":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_three.ttm'))
        if self.args.load_type == "four":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_four.ttm'))
        if self.args.load_type == "six":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_six.ttm'))
        ####################
        
        self.payload = Shape('Cuboid4')
        self.payload.set_orientation([0.0,0.0,0.0])
        self.payload.set_position([0.0,0.0,0.1])

        return m2


    def generate_goal(self):
        #####################################
        #visualization goal
        if self.args.load_type == "three":
            self.payload_1 = Shape('Cuboid28')
            self.payload_2 = Shape('Cuboid29')
            self.payload_3 = Shape('Cuboid30')

            loads = [self.payload_1,self.payload_2,self.payload_3]

        if self.args.load_type == "four":
            self.payload_1 = Shape('Cuboid24')
            self.payload_2 = Shape('Cuboid25')
            self.payload_3 = Shape('Cuboid28')
            self.payload_4 = Shape('Cuboid29')

            loads = [self.payload_1,self.payload_2,self.payload_3,self.payload_4]
        
        if self.args.load_type == "six":
            self.payload_1 = Shape('Cuboid24')
            self.payload_2 = Shape('Cuboid25')
            self.payload_3 = Shape('Cuboid26')
            self.payload_4 = Shape('Cuboid27')
            self.payload_5 = Shape('Cuboid28')
            self.payload_6 = Shape('Cuboid29')

            loads = [self.payload_1,self.payload_2,self.payload_3,self.payload_4,self.payload_5,self.payload_6] 

        points = []

        if not self.is_pc:
            for i in range(len(loads)):
                points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])
        else:
            if self.num_a == 3:
                assert(len(loads)==6)
                choice_list = random.sample(range(len(loads)),k=3)
                # choice_list.sort() # permutation invariance
                for i in range(len(choice_list)):
                    points.append([loads[choice_list[i]].get_position()[0],loads[choice_list[i]].get_position()[1],1.5])
            if self.num_a == 6:
                assert(len(loads)==6)
                for i in range(len(loads)):
                    points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])
            
        goal_points = np.array(points)

        for i in range(self.num_a):
            self.targets[i].set_position(goal_points[i])

        return goal_points
    

    def random_position_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        saved_agents = []
        vpts = []
        for i in range(self.num_a):
            if self.is_sensor_obs:
                [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL.ttm'))
            else:
                [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)
            objs.append(m)
            if self.is_sensor_obs:
                self.agents.append(Drone_s_w(i))
            else:
                self.agents.append(Drone_s(i))
            if i == 0:
                self.agents[i].agent.set_3d_pose([random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max),1.7,0.0,0.0,0.0])
                vx = self.agents[i].agent.get_position()[0]
                vy = self.agents[i].agent.get_position()[1]
                vpts.append([vx,vy])
                saved_agents.append(i)
            else:
                vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                check_list = [self.check_collision_p(vpt,vpts[m]) for m in saved_agents]
                check_conditions = np.sum(check_list)

                while check_conditions:
                    vpt = [random.uniform(self.x_limit_min,self.x_limit_max),random.uniform(self.y_limit_min,self.y_limit_max)]
                    check_list = [self.check_collision_p(vpt,vpts[m]) for m in saved_agents]
                    check_conditions = np.sum(check_list)

                # print("all",vpts)
                # print("current",vpt)
                self.agents[i].agent.set_3d_pose([vpt[0],vpt[1],1.7,0.0,0.0,0.0])
                vpts.append(vpt)
                saved_agents.append(i)
        return model_handles,objs

    def direct_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"

        if self.num_a == 3:
            pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)
        
        if self.num_a == 4:
            pos = np.array([[3.5,4.5],[3.5,-4.5],[-3.5,-4.5],[-3.5,4.5]])
            # pos = np.array([[1.5,4.5],[3.5,4.5],[-1.5,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        if self.num_a == 6:
            pos = np.array([[4.5,4.5],[4.5,-4.5],[-4.5,4.5],[-4.5,-4.5],[0,4.5],[0,-4.5]])
            # pos = np.array([[3.5,4.5],[3.5,-4.5],[-3.5,4.5],[-3.5,-4.5],[0,4.5],[0,-4.5]])
            # pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        for i in range(self.num_a):
            if self.is_sensor_obs:
                [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL.ttm'))
            else:
                [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)
            objs.append(m)
            if self.is_sensor_obs:
                self.agents.append(Drone_s_w(i))
            else:
                self.agents.append(Drone_s(i))
            
            self.agents[i].agent.set_3d_pose([pos[i,0],pos[i,1],1.7,0.0,0.0,0.0])
                

        return model_handles,objs

    def reset_world(self):
        self.time_step = 0
        
        #self.suction_cup.release()
        if not self.close_simulation:
            for i in range(self.num_a):
                self.pr.remove_model(self.model_handles[i])
            self.pr.remove_model(self.load_handle)
        
        self.model_handles,ii = self.random_position_spread()
        # self.model_handles,ii = self.direct_spread()
        
        self.load_handle = self.load_spread()

        if self.close_simulation:
            self.targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.56, 0.56, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_a)]

        self.goals = self.generate_goal()

        for j in range(self.num_a):
            self.agents[j]._reset()
    
        #for hovering when reset
        for j in range(50):
            for agent in self.agents:
                agent.hover(1.7)
            self.pr.step()
     
        obs_n = []
        for agent in self.agents:
            obs_n.append(self.observation_callback(agent))

        self.close_simulation = False

        return np.array(obs_n)


    def step(self, action_n):

        obs_n = []
        reward_n = []
        done_n = []
        done_ter = []

        for j in range(10):
            # set action for each agent
            for i, agent in enumerate(self.agents):
                agent.set_action(action_n[i])
            self.pr.step()

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            #----------------------------
            obs_n.append(self.observation_callback(agent))
            rw,ter,dter = self.reward_callback(agent)
            reward_n.append(rw)
            done_n.append(ter)
            done_ter.append(dter)
            
        #all agents get total reward in cooperative case
        reward = reward_n[0] #need modify

        if np.all(done_ter):
            reward = 1*self.num_a

        #once collision every agent will be pulished
        if np.any(done_n):
            #reward = -50
            reward = reward - 1*self.num_a

        done_all = [np.any(done_n)]* self.num_a

        if self.shared_reward:
            reward_n = [reward] * self.num_a

        self.time_step+=1

        return np.array(obs_n), np.array(reward_n), np.array(done_all), np.all(done_ter)


    def step_evaluate(self, action_n):

        obs_n = []
        reward_n = []
        done_n = []
        done_ter = []

        for j in range(10):
            # set action for each agent
            for i, agent in enumerate(self.agents):
                agent.set_action(action_n[i])
            self.pr.step()

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            #----------------------------
            obs_n.append(self.observation_callback(agent))
            rw,ter,dter = self.reward_callback(agent)
            reward_n.append(rw)
            done_n.append(ter)
            done_ter.append(dter)
            
        #all agents get total reward in cooperative case
        reward = reward_n[0] #need modify

        if np.all(done_ter):
            reward = 1*self.num_a

        #once collision every agent will be pulished
        if np.any(done_n):
            #reward = -50
            reward = reward - 1*self.num_a

        done_all = [np.any(done_n)]* self.num_a

        if self.shared_reward:
            reward_n = [reward] * self.num_a

        self.time_step+=1

        return np.array(obs_n), np.array(reward_n), np.array(done_all), np.all(done_ter)


    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def restart(self):
        if self.pr.running:
            self.pr.stop()
            
        self.pr.shutdown()

        self.pr = PyRep()
        self.pr.launch(self.env_name, headless=False)
        self.pr.start()
        self.close_simulation = True
        

    # def reward_and_terminate(self, agent):
    #     rew = 0
    #     done_terminate = 0
    #     terminate = 0
    #     finish_sig = np.zeros(self.num_a)

    #     max_roll = 1.57 # Max roll after which we end the episode
    #     max_pitch = 1.57 # Max roll after which we end the episode

    #     current_orientation = agent.agent.get_orientation()

    #     has_flipped = True
    #     if current_orientation[0] > -1*max_roll and current_orientation[0] <= max_roll:
    #         if current_orientation[1] > -1*max_pitch and current_orientation[1] <= max_pitch:
    #             has_flipped = False

    #     #team reward
    #     for i in range(self.goals.shape[0]):
    #         dists = [np.sqrt(np.sum(np.square(a.get_2d_pos() - self.goals[i,:2]))) for a in self.agents]
    #         finish_sig[i] = np.any((np.array(dists)<0.5))
    #         rew -= min(dists)/(self.field_size*2)
            
    #     if np.all(finish_sig):
    #         done_terminate = 1 

    #     #collision detection
    #     wall_dists = np.array([np.abs(self.field_size-agent.agent.get_position()[1]),np.abs(self.field_size+agent.agent.get_position()[1]),np.abs(self.field_size+agent.agent.get_position()[0]),np.abs(self.field_size-agent.agent.get_position()[0])]) # rangefinder: forward, back, left, right
    #     wall_sig = np.any(wall_dists<0.206)

    #     agent_collision = []
    #     for a in self.agents:
    #         if a == agent: continue
    #         if self.check_collision_a(agent,a):
    #             agent_collision.append(1)
    #         else:
    #             agent_collision.append(0)
    #     agent_sig = np.any(np.array(agent_collision))

    #     if agent_sig or wall_sig or has_flipped:
    #         terminate = 1

    #     if agent.agent.get_position()[2]<1.3:
    #         terminate = 1

    #     return rew,terminate,done_terminate

    def reward_and_terminate(self, agent):
        rew = 0
        done_terminate = 0
        terminate = 0
        finish_sig = np.zeros(self.num_a)

        max_roll = 1.57 # Max roll after which we end the episode
        max_pitch = 1.57 # Max roll after which we end the episode

        current_orientation = agent.agent.get_orientation()

        has_flipped = True
        if current_orientation[0] > -1*max_roll and current_orientation[0] <= max_roll:
            if current_orientation[1] > -1*max_pitch and current_orientation[1] <= max_pitch:
                has_flipped = False

        # each column represents distance of all agents from the respective landmark
        world_dists = np.array([[np.linalg.norm(a.get_2d_pos() - self.goals[l,:2]) for l in range(self.goals.shape[0])]
                               	for a in self.agents])
        # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
        ri, ci = linear_sum_assignment(world_dists)
        min_dists = world_dists[ri, ci]
        rew = -np.mean(min_dists)/(self.field_size*2)

        #team reward
        for i in range(self.goals.shape[0]):
            dists = [np.sqrt(np.sum(np.square(a.get_2d_pos() - self.goals[i,:2]))) for a in self.agents]
            finish_sig[i] = np.any((np.array(dists)<0.5))
            
        if np.all(finish_sig):
            done_terminate = 1 

        #collision detection
        wall_dists = np.array([np.abs(self.field_size-agent.agent.get_position()[1]),np.abs(self.field_size+agent.agent.get_position()[1]),np.abs(self.field_size+agent.agent.get_position()[0]),np.abs(self.field_size-agent.agent.get_position()[0])]) # rangefinder: forward, back, left, right
        wall_sig = np.any(wall_dists<0.206)

        agent_collision = []
        for a in self.agents:
            if a == agent: continue
            if self.check_collision_a(agent,a):
                agent_collision.append(1)
            else:
                agent_collision.append(0)
        agent_sig = np.any(np.array(agent_collision))

        if agent_sig or wall_sig or has_flipped:
            terminate = 1

        if agent.agent.get_position()[2]<1.3:
            terminate = 1

        return rew,terminate,done_terminate


    def observation(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i in range(self.goals.shape[0]):  # world.entities:
            entity_pos.append((self.goals[i,:2]-agent.get_2d_pos())/(self.field_size*2))   
        
        # communication of all other agents
        other_pos = []
        for other in self.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append((other.get_2d_pos() - agent.get_2d_pos())/(self.field_size*2))

        other_vel = []
        for other in self.agents:
            if other is agent: continue
            other_vel.append(other.get_2d_vel())

        return np.concatenate([agent.get_2d_vel()]+ [agent.get_2d_pos()/self.field_size] + other_vel + other_pos + entity_pos)

    
    #local observation 
    def local_observation(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i in range(self.goals.shape[0]):  # world.entities:
            distance = np.sqrt(np.sum(np.square(self.goals[i,:2]-agent.get_2d_pos())))
            if distance>self.sight:
                entity_pos.append([0,0,0])
            else:
                entity_pos.append((self.goals[i,:2]-agent.get_2d_pos())/(self.field_size*2)) 
                entity_pos.append([1])  
        
        # communication of all other agents
        other_live = []
        other_pos = []
        other_vel = []
        for other in self.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            distance = np.sqrt(np.sum(np.square(other.get_2d_pos()-agent.get_2d_pos())))
            if distance>self.sight:
                other_pos.append([0,0])
                other_vel.append([0,0])
                other_live.append([0])
            else:
                other_pos.append((other.get_2d_pos() - agent.get_2d_pos())/(self.field_size*2))
                other_vel.append(other.get_2d_vel())
                other_live.append([1])

        return np.concatenate([agent.get_2d_vel()]+ [agent.get_2d_pos()/self.field_size] + other_vel + other_pos + other_live + entity_pos)

    def sensor_observation(self,agent):
        # get positions of all entities in this agent's reference frame
        if self.num_a == 3:
            goals = np.array([[2.8,0,0.4],[-2.8,0,0.4],[0,0,0.4]])
        if self.num_a == 4:
            goals = np.array([[1.25,1.25,0.4],[-1.25,1.25,0.4],[1.25,-1.25,0.4],[-1.25,-1.25,0.4]])
        if self.num_a == 6:
            goals = np.array([[2.25,-2.25,0.4],[-2.25,-2.25,0.4],[1.2,0,0.4],[-1.2,0,0.4],[2.25,2.25,0.4],[-2.25,2.25,0.4]])
        
        entity_pos = (agent.get_goals_sensor(goals)-(agent.agent.get_drone_position()[2]-0.4))/np.sqrt((agent.agent.get_drone_position()[2]-0.4)**2+(self.field_size*2)**2)
        # for i in range(self.goals.shape[0]):  # world.entities:
        #     entity_pos.append((self.goals[i,:2]-agent.get_2d_pos())/(self.field_size*2))     
        
        # communication of all other agents
        other_pos = []
        o_pos = agent.get_neighbour_pos_from_camera()
        for i in range(o_pos.shape[0]):
            other_pos.append((o_pos[i,:2])/(self.field_size*2))

        # print(len(other_pos))
        assert((self.num_a-1)>=len(other_pos))
        for _ in range(self.num_a-1-len(other_pos)):
            other_pos.append(np.zeros(2))
        
        assert(len(other_pos)==(self.num_a-1))


        # other_vel = []
        # for other in self.agents:
        #     if other is agent: continue
        #     other_vel.append(other.get_2d_vel())
        # print(entity_pos)
        # print(other_pos)

        return np.concatenate([agent.get_2d_vel()]+ [agent.get_2d_pos()/self.field_size] + other_pos + [np.array(entity_pos)])
        


