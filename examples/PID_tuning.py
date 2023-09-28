from os import path
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.mobiles.new_quadricopter import NewQuadricopter
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
from pyrep.policies.maddpg_drone_agent_scratch import Agent


LOOPS = 1
SCENE_FILE = join(dirname(abspath(__file__)), 'PID_tune.ttt')

pr = PyRep()  
pr.launch(SCENE_FILE, headless=False)
# print(pr.get_simulation_timestep())
pr.step_ui()
pr.start()

# robot_DIR = path.join(path.dirname(path.abspath(__file__)), 'models')

# #pr.remove_model(m1)
# [m,m1]= pr.import_model(path.join(robot_DIR, 'bacterium_drone1.ttm'))

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
# print(sim.simGetSimulationTimeStep())
for i in range(LOOPS):
  agent1 = NewQuadricopter(0,4)
  agent2 = NewQuadricopter(1,4)
  agent3 = NewQuadricopter(1,4)
  agent4 = NewQuadricopter(2,4)
  agent5 = NewQuadricopter(3,4)
  agent6 = NewQuadricopter(4,4)
  agent7 = NewQuadricopter(5,4)
  agent8 = NewQuadricopter(6,4)
  agent9 = NewQuadricopter(7,4)
  agent10 = NewQuadricopter(8,4)
  agent11 = NewQuadricopter(9,4)
  agent12 = NewQuadricopter(10,4)
  
 

  # Get a random position within a cuboid and set the target position
  #suction_cup.release()

  # pr.remove_model(m1)
  # [m,m1]= pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))

  #agent.drone_reset()
  #agent.set_3d_pose([0.0,0.0,1.5,0.0,0.0,0.0])

  # agent1.set_drone_position([0.0,0.0,1.2])
  # vx = agent1.get_drone_position()[0]
  # vy = agent1.get_drone_position()[1]

  for i in range(250):
    agent1.hover1()
    agent2.hover1()
    agent3.hover1()
    agent4.hover1()
    agent5.hover1()
    agent6.hover1()
    agent7.hover1()
    agent8.hover1()
    agent9.hover1()
    agent10.hover1()
    agent11.hover1()
    agent12.hover1()

    pr.step()
 
  #set_position([0.0,0.0,0.50])
  # target.set_orientation([0.0, 0.0, 0.0])
  
  for j in np.arange(3000):
      # agent1.panoramic_camera.handle_explicitly()
      # print(agent1.get_concentration())
      img = agent1.panoramic_camera.capture_rgb()
      # imageio.imwrite('wawa.png',img)
      # print(img.min(),img.max())
      
      # if j<300:
      #   vels = agent1.velocity_controller_xy(0.1,0.1,1.0)
      # else:
      # vels = agent1.velocity_controller_xy(0.1,0.1,1.5)
      # # else: 
      # #   vels = agent1.velocity_controller(0.1,0.1)
      #   # vels = agent1.position_controller([0.5,0.5,0.9])
      #   # vels = agent1.velocity_controller(0.1,0.1)

      #   #print(vels[0], vels[1], vels[2], vels[3])
      # agent1.set_propller_velocity([vels[0], vels[1], vels[2], vels[3]])

      #   # Power each propeller
      # agent1.control_propeller_thrust(1)
      # agent1.control_propeller_thrust(2)
      # agent1.control_propeller_thrust(3)
      # agent1.control_propeller_thrust(4)
      action = np.array([0.5,0.0,0.5])


      vels = agent1.velocity_controller1(action[0],action[1])

      agent1.set_propller_velocity(vels[:])

      agent1.control_propeller_thrust(1)
      agent1.control_propeller_thrust(2)
      agent1.control_propeller_thrust(3)
      agent1.control_propeller_thrust(4)


      vels = agent2.velocity_controller1(action[0],action[1])

      agent2.set_propller_velocity(vels[:])

      agent2.control_propeller_thrust(1)
      agent2.control_propeller_thrust(2)
      agent2.control_propeller_thrust(3)
      agent2.control_propeller_thrust(4)

      vels = agent3.velocity_controller1(action[0],action[1])

      agent3.set_propller_velocity(vels[:])

      agent3.control_propeller_thrust(1)
      agent3.control_propeller_thrust(2)
      agent3.control_propeller_thrust(3)
      agent1.control_propeller_thrust(4)



    #   pr.step()
    #   z1 = agent1.get_drone_position()[2]

    #   vels = agent1.velocity_controller_xy(action[1],action[2],action[0])
    #   agent1.set_propller_velocity(vels[:])

    #   agent1.control_propeller_thrust(1)
    #   agent1.control_propeller_thrust(2)
    #   agent1.control_propeller_thrust(3)
    #   agent1.control_propeller_thrust(4)

      #det = suction_cup.grasp(target)
    
      pr.step()
      
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

