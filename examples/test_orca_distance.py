# from os.path import dirname, join, abspath
# from pyrep import PyRep
# from pyrep.robots.arms.panda import Panda
# from pyrep.envs.drone_RL_agent import Drone

# SCENE_FILE = join(dirname(abspath(__file__)), 'scene_panda_reach_target.ttt')
# DELTA = 0.01
# pr = PyRep()
# pr.launch(SCENE_FILE, headless=False)
# pr.start()
# agent = Drone(0).agent

# for obj in agent.get_objects_in_tree():
#     print(f'{obj.get_name()}: {obj}')

# pr.stop()
# pr.shutdown()
import torch

def ORCA_distance(vel, points):
        #input is array, vel is tensor
        point = vel
        line_point1 = points[:2]
        line_point2 = points[:2]+points[2:4]

        vec1 = line_point1-point
        vec2 = line_point2-point

        vec1_3d = torch.zeros(3)
        vec2_3d = torch.zeros(3)
        vec1_3d[0] = vec1[0]
        vec1_3d[1] = vec1[1]
        vec2_3d[0] = vec2[0]
        vec2_3d[1] = vec2[1]

        distance = torch.abs(torch.cross(vec1_3d,vec2_3d)[2])/torch.linalg.norm(line_point1-line_point2)

        return distance

a = torch.tensor([0.0,0.0])
b = torch.tensor([1.0,0.0,-1.0,1.0])
print(ORCA_distance(a,b))