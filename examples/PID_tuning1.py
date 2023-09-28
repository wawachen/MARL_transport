from os import path
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.envs.drone_RL_agent import Drone
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import PrimitiveShape
from pyrep.robots.end_effectors.uarm_Vacuum_Gripper import UarmVacuumGripper
import numpy as np
from multiprocessing import Process,Value
from cv2 import VideoWriter, VideoWriter_fourcc
import matplotlib.pyplot as plt
import cv2
import random
import math
from pyrep.backend import sim
import imageio


LOOPS = 1
SCENE_FILE = join(dirname(abspath(__file__)), 'PID_tune.ttt')

pr = PyRep()  
pr.launch(SCENE_FILE, headless=False)
pr.set_simulation_timestep(10.0)
pr.step_ui()
pr.start()

robot_DIR = path.join(path.dirname(path.abspath(__file__)), 'models')

#pr.remove_model(m1)
[m,m1]= pr.import_model(path.join(robot_DIR, 'new_drone.ttm'))

#pr.remove_model(m1)img.min() == img.max() == 0.0
# agent1 = NewQuadricopter(0,4)
# agent2 = NewQuadricopter(1,4)
# agent3 = NewQuadricopter(2,4)
# agent4 = NewQuadricopter(3,4)
# agent5 = NewQuadricopter(4,4)
# agent6 = NewQuadricopter(5,4)

# target = Shape('Payload')
#suction_cup = UarmVacuumGripper()

# We could have made this target in the scene, but lets create one dynamically
#target = Shape.create(type=PrimitiveShape.CUBOID,
#                      size=[1.2, 1.2, 0.8],
#                      color=[0.4, 0.8, 0.1],
#                      static=False, respondable=True)
print(sim.simGetSimulationTimeStep())
for i in range(LOOPS):
  agent1 = Drone(0)
 

  # Get a random position within a cuboid and set the target position
  #suction_cup.release()

  # pr.remove_model(m1)
  # [m,m1]= pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))

  #agent.drone_reset()
  #agent.set_3d_pose([0.0,0.0,1.5,0.0,0.0,0.0])

  agent1.agent.set_drone_position([0.0,0.0,1.7])
#   vx = agent1.get_2d_pos()[0]
#   vy = agent1.get_2d_pos()[1]
 
  #set_position([0.0,0.0,0.50])
  # target.set_orientation([0.0, 0.0, 0.0])
  for _ in range(100):
    pos = agent1.get_2d_pos()[:]
    agent1.hover(pos)
    pr.step()
  
  print(agent1.get_2d_pos()[:])

  for j in np.arange(100):
      # agent1.panoramic_camera.handle_explicitly()
      #img = agent1.panoramic_camera.capture_rgb()
      # imageio.imwrite('wawa.png',img)
      # print(img.min(),img.max())
      if j<10:
        agent1.set_action_pos(np.array([6,1]),pr)
        print(agent1.get_2d_pos()[:])
      elif j<20:
        agent1.set_action_pos(np.array([7,0]),pr)
        print(agent1.get_2d_pos()[:])
      elif j<30:
        agent1.set_action_pos(np.array([-9,1]),pr)
        print(agent1.get_2d_pos()[:])
      else:
        agent1.set_action_pos(np.array([-7,-1]),pr)
        print(agent1.get_2d_pos()[:])
      # agent1.set_action(np.array([1,1]))
      # agent1.set_action_pos(np.array([0.7,-0.2]),pr)
      # if j<100:
      #   agent1.set_action(np.array([1,1]))
      # elif j<150:
      #   pos = agent1.get_2d_pos()[:]
      #   agent1.hover(pos)
      # elif j<250:
      #   agent1.set_action(np.array([1,0]))
      # elif j<300:
      #   pos = agent1.get_2d_pos()[:]
      #   agent1.hover(pos)
      # elif j<350:
      #   agent1.set_action(np.array([-1,1]))
      # elif j<400:
      #   pos = agent1.get_2d_pos()[:]
      #   agent1.hover(pos)
      # else:
      #   agent1.set_action(np.array([-1,-1]))
      # else: 
      #   vels = agent1.velocity_controller(0.1,0.1)
        # vels = agent1.position_controller([0.5,0.5,0.9])
        # vels = agent1.velocity_controller(0.1,0.1)

        #print(vels[0], vels[1], vels[2], vels[3])

      #det = suction_cup.grasp(target)
    
      # pr.step()
      
      #print(np.r_[(target.get_position()[:]-np.array([0.0,0.0,1.5])),target.get_orientation()[:],agent.get_drone_position()[:], agent.velocities])    
     


pr.stop()
pr.shutdown()



#robot_DIR = path.join(path.dirname(path.abspath(__file__)), 'models')
#m = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))
#m1 = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))
#m2 = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))
#m3 = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))

'''cam = VisionSensor('my_cam')
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

width = 1280
height = 720
fps = 24
fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./my_vid.avi', fourcc, float(fps), (width, height))

pr = PyRep()
pr.launch(...)
cam = VisionSensor('my_cam')

for _ in range(100):
    img = (cam.capture_rgb() * 255).astype(np.uint8)
    video.write(img)
video.release()
'''

