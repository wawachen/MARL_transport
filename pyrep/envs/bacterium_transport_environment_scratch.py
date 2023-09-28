from os import path
from pyrep import PyRep
from pyrep.envs.drone_RL_agent import Drone_s
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
import random
import math
from gym import spaces

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Drone_Env:

    def __init__(self, args, env_name,num_agents):
        self.args = args
        self.reset_callback = self.reset_world
        self.reward_callback = self.reward_and_terminate
        self.observation_callback = self.observation
        # self.done_callback = self.done

        self.is_pc = self.args.is_pc
        self.is_local_obs = self.args.is_local_obs

        # environment parameters
        self.discrete_action_space = False
        self.time_step = 0
        self.field_size = self.args.field_size/2 #field_size can 10 or 15

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.num_a = num_agents
        self.env_name = env_name
        # self.close_simulation = False

        self.safe_distance = 0.71
        self.x_limit_min = -self.field_size+self.safe_distance/2
        self.x_limit_max = self.field_size-self.safe_distance/2
        self.y_limit_min = -self.field_size+self.safe_distance/2
        self.y_limit_max = self.field_size-self.safe_distance/2

        self.shared_reward = True
        
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        
        self.model_handles, self.load_handle = self.import_agent_models()
        self.agents = [Drone_s(i) for i in range(num_agents)]
        # self.reset_world()
        self.payload = Shape('Cuboid4')
        self.targets = [Shape.create(type=PrimitiveShape.CYLINDER,
                      size=[0.56, 0.56, 0.01],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False) for i in range(self.num_a)]

        self.goals = self.generate_goal()

        self.enterlift = 0
        self.enterhover = 0
        self.pos_des = np.zeros([num_agents,3])
        self.pos_des1 = np.zeros([num_agents,3])
        self.concen_collect = np.zeros(num_agents)

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
            # agent.action.c = np.zeros(self.world.dim_c)

    def import_agent_models(self):
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        
        #pr.remove_model(m1)
        model_handles = []

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)

        if self.args.load_type == "three":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_three.ttm'))
        if self.args.load_type == "four":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_four.ttm'))
        if self.args.load_type == "six":
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_six_transport.ttm'))

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
            [m22,m2]= self.pr.import_model(path.join(robot_DIR, 'load_six_transport.ttm'))
        ####################
        
        self.payload = Shape('Cuboid4')
        self.payload.set_orientation([0.0,0.0,0.0])
        self.payload.set_position([0.0,0.0,0.3])

        return m2

    def generate_goal(self):
        #####################################
        #visualization goal
        # if self.args.load_type == "three":
        #     self.payload_1 = Shape('Cuboid28')
        #     self.payload_2 = Shape('Cuboid29')
        #     self.payload_3 = Shape('Cuboid30')

        #     loads = [self.payload_1,self.payload_2,self.payload_3]

        # if self.args.load_type == "four":
        #     self.payload_1 = Shape('Cuboid24')
        #     self.payload_2 = Shape('Cuboid25')
        #     self.payload_3 = Shape('Cuboid28')
        #     self.payload_4 = Shape('Cuboid29')

        #     loads = [self.payload_1,self.payload_2,self.payload_3,self.payload_4]
        
        # if self.args.load_type == "six":
        #     self.payload_1 = Shape('Cuboid24')
        #     self.payload_2 = Shape('Cuboid25')
        #     self.payload_3 = Shape('Cuboid26')
        #     self.payload_4 = Shape('Cuboid27')
        #     self.payload_5 = Shape('Cuboid28')
        #     self.payload_6 = Shape('Cuboid29')

        #     loads = [self.payload_1,self.payload_2,self.payload_3,self.payload_4,self.payload_5,self.payload_6] 

        # points = []

        # if not self.is_pc:
        #     for i in range(len(loads)):
        #         points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])
        # else:
        #     if self.num_a == 3:
        #         assert(len(loads)==6)
        #         choice_list = random.sample(range(len(loads)),k=3)
        #         # choice_list.sort() # permutation invariance
        #         for i in range(len(choice_list)):
        #             points.append([loads[choice_list[i]].get_position()[0],loads[choice_list[i]].get_position()[1],1.5])
        #     if self.num_a == 6:
        #         assert(len(loads)==6)
        #         for i in range(len(loads)):
        #             points.append([loads[i].get_position()[0],loads[i].get_position()[1],1.5])
            
        goal_points = np.array([[2.25,2.25,1.5],[-2.25,2.25,1.5],[1.2,0,1.5],[-1.2,0,1.5],[-2.25,-2.25,1.5],[2.25,-2.25,1.5]])

        for i in range(self.num_a):
            self.targets[i].set_position(goal_points[i])

        return goal_points

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
            # pos = np.array([[2.5,4.5],[3.5,4.5],[-2.5,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        if self.num_a == 6:
            pos = np.array([[4.5,4.5],[4.5,-4.5],[-4.5,4.5],[-4.5,-4.5],[0,4.5],[0,-4.5]])
            # pos = np.array([[0.0,4.5],[3.5,-4.5],[-3.5,-4.5]])
            # pos = np.array([[3.5,4.5],[0,4.5],[-3.5,4.5]])
            assert(pos.shape[0]==self.num_a)

        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)
            objs.append(m)
            self.agents.append(Drone_s(i))
            
            self.agents[i].agent.set_3d_pose([pos[i,0],pos[i,1],1.7,0.0,0.0,0.0])
                

        return model_handles,objs

    def random_position_spread(self):
        self.agents = []
        model_handles = []
        objs = []
        robot_DIR = "/home/wawa/RL_transport_3D/examples/models"
        saved_agents = []
        vpts = []
        for i in range(self.num_a):
            [m,m1]= self.pr.import_model(path.join(robot_DIR, 'bacterium_drone_RL_state.ttm'))
            model_handles.append(m1)
            objs.append(m)
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

    
    def reset_world(self):
        self.time_step = 0
        
        #self.suction_cup.release()
        for i in range(self.num_a):
            self.pr.remove_model(self.model_handles[i])
        self.pr.remove_model(self.load_handle)
        # self.model_handles = self.import_agent_models()
        # # self.agents = [Drone(i) for i in range(self.num_a)]
        # self.random_position_spread()

        # self.model_handles,ii = self.random_position_spread()
        self.model_handles,ii = self.direct_spread()
        # self.goals = self.generate_goal()

        self.load_handle = self.load_spread()

        self.goals = self.generate_goal()

        for j in range(self.num_a):
            self.agents[j]._reset()
    
        #for hovering when reset
        for j in range(120):
            for agent in self.agents:
                agent.hover(1.7)
            self.pr.step()
     
        obs_n = []
        for agent in self.agents:
            obs_n.append(self.observation_callback(agent))


        return np.array(obs_n)


    def step(self, action_n):
        finish_sig = np.zeros(self.num_a)
        for i in range(self.goals.shape[0]):
            dists = [np.sqrt(np.sum(np.square(a.get_2d_pos() - self.goals[i,:2]))) for a in self.agents]
            finish_sig[i] = np.any((np.array(dists)<0.5))
        
        if np.all(finish_sig) or self.enterhover:
            #---------------transport------------------------------
            for i in range(self.num_a):
                detect = self.agents[i].agent.suction_cup.grasp(self.payload)
        
            if self.time_step > 200:
                if self.enterlift == 0:
                    for i in range(self.num_a):
                        p_pos = self.agents[i].agent.get_drone_position()
                        self.pos_des1[i,0] = p_pos[0]
                        self.pos_des1[i,1] = p_pos[1]
                        self.pos_des1[i,2] = 2.0

                ee = np.zeros(self.num_a)
                for i in range(self.num_a):
                    # flock_vel = self.agents[i].flock_controller()
                    vels = self.agents[i].agent.position_controller1(self.pos_des1[i,:])

                    self.agents[i].agent.set_propller_velocity(vels)

                    self.agents[i].agent.control_propeller_thrust(1)
                    self.agents[i].agent.control_propeller_thrust(2)
                    self.agents[i].agent.control_propeller_thrust(3)
                    self.agents[i].agent.control_propeller_thrust(4)
                
                self.enterlift = 1

            else:
                if self.time_step<100:
                    for agent in self.agents:
                        agent.hover(1.7)
                else:
                    if self.enterhover == 0:
                        for i in range(self.num_a):
                            p_pos = self.agents[i].agent.get_drone_position()
                            obj_h = self.agents[i].agent.get_concentration(self.payload)
                            self.concen_collect[i] = obj_h
                            des_h = p_pos[2] - (self.agents[i].agent.suction_cup.get_suction_position()[2]-obj_h)+0.0565+0.02+0.0019-0.01
                            self.pos_des[i,0] = p_pos[0]
                            self.pos_des[i,1] = p_pos[1]
                            self.pos_des[i,2] = des_h
                            print("agent",i,": ",obj_h)
                    self.enterhover = 1
                    for i in range(self.num_a):
                        # flock_vel = self.agents[i].flock_controller()
                        vels = self.agents[i].agent.position_controller1(self.pos_des[i,:])
                        self.agents[i].agent.set_propller_velocity(vels)
                        self.agents[i].agent.control_propeller_thrust(1)
                        self.agents[i].agent.control_propeller_thrust(2)
                        self.agents[i].agent.control_propeller_thrust(3)
                        self.agents[i].agent.control_propeller_thrust(4)

            # print("time",self.time_step)
            self.pr.step()
            self.time_step+=1
            obs_n = []
            for i, agent in enumerate(self.agents):
                #----------------------------
                obs_n.append(np.zeros(self.num_a*2+(self.num_a-1)*4))

            return np.array(obs_n),0, 0
            #------------------------------------------------------
        else:
            obs_n = []
            reward_n = []
            done_n = []
            done_ter = []

            for j in range(10):
                # set action for each agent
                for i, agent in enumerate(self.agents):
                    agent.set_action(action_n[i])
                    # agent.set_action_pos(action_n[i], self.pr)
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
                reward = 0
                reward = reward + 1*self.num_a

            #once collision every agent will be pulished
            if np.any(done_n):
                reward = reward - 1*self.num_a

            if self.shared_reward:
                reward_n = [reward] * self.num_a

            self.time_step+=1

        return np.array(obs_n), np.array(reward_n), np.array(done_n)


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
        # self.close_simulation = True
        

    def reward_and_terminate(self, agent):
        rew = 0
        done_terminate = 0
        terminate = 0
        finish_sig = np.zeros(self.num_a)

        #team reward
        for i in range(self.goals.shape[0]):
            dists = [np.sqrt(np.sum(np.square(a.get_2d_pos() - self.goals[i,:2]))) for a in self.agents]
            finish_sig[i] = np.any((np.array(dists)<0.5))
            rew -= min(dists)/(self.field_size*2)
            
        
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

        if agent_sig or wall_sig:
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



