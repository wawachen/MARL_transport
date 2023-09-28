from pyrep.robots.mobiles.turtlebot import TurtleBot
from pyrep.robots.mobiles.turtlebot_original import TurtleBot_o
import numpy as np
from pyrep.envs.array_specs import ArraySpec

class Turtle:
    def __init__(self, id):
        self.agent = TurtleBot(id)

    def get_2d_pos(self):
        return self.agent.get_position()[:2]

    def get_heading(self):
        return self.agent.get_orientation()[2]

    def is_crash(self):
        return self.agent.check_collision() # check whether collide with all objects
        
    # def _reset(self):
    #     p = np.random.random.uniform(-2.5,2.5)
    #     self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])

    def set_action(self, action):
        #Turtlebot is controlled by angular velocity in z axis and linear velocity in x axis
        actuation = self.agent.move(action)
        self.agent.set_joint_target_velocities(actuation)


class Turtle_o:
    def __init__(self, id):
        self.agent = TurtleBot_o(id)

    def get_2d_pos(self):
        return self.agent.get_position()[:2]

    def get_heading(self):
        return self.agent.get_orientation()[2]

    def is_crash(self):
        return self.agent.check_collision() # check whether collide with all objects
        
    # def _reset(self):
    #     p = np.random.random.uniform(-2.5,2.5)
    #     self.agent.set_3d_pose([p[0],p[1],1.7,0.0,0.0,0.0])

    def set_action(self, action):
        #Turtlebot is controlled by angular velocity in z axis and linear velocity in x axis
        actuation = self.agent.move(action)
        self.agent.set_joint_target_velocities(actuation)

   
            
        
