from os import path
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.mobiles.new_quadricopter import NewQuadricopter
import numpy as np
from mpi_test import run
from mpi4py import MPI

# def run(comm, env):

if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  LOOPS = 1
  SCENE_FILE = join(dirname(abspath(__file__)), 'PID_tune.ttt')

  pr = PyRep()  
  pr.launch(SCENE_FILE, headless=False)
  # print(pr.get_simulation_timestep())
  pr.step_ui()
  pr.start()


  for i in range(LOOPS):
    agent1 = NewQuadricopter(0,4)
  

    for i in range(250):
      agent1.hover1()
      pr.step()
    
    for j in np.arange(3000):
        action = np.array([0.5,0.5,0.5])

        vels = agent1.velocity_controller1(action[0],action[1])

        agent1.set_propller_velocity(vels[:])

        agent1.control_propeller_thrust(1)
        agent1.control_propeller_thrust(2)
        agent1.control_propeller_thrust(3)
        agent1.control_propeller_thrust(4)
      
        pr.step()  

  pr.stop()
  pr.shutdown()



# #robot_DIR = path.join(path.dirname(path.abspath(__file__)), 'models')
# #m = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))
# #m1 = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))
# #m2 = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))
# #m3 = pr.import_model(path.join(robot_DIR, 'Quadricopter_rope.ttm'))

# '''cam = VisionSensor('my_cam')
# import numpy as np
# from cv2 import VideoWriter, VideoWriter_fourcc

# width = 1280
# height = 720
# fps = 24
# fourcc = VideoWriter_fourcc(*'MP42')
# video = VideoWriter('./my_vid.avi', fourcc, float(fps), (width, height))

# pr = PyRep()
# pr.launch(...)
# cam = VisionSensor('my_cam')

# for _ in range(100):
#     img = (cam.capture_rgb() * 255).astype(np.uint8)
#     video.write(img)
# video.release()
# '''
 
# from mpi4py import MPI
# import numpy as np
# import sys

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()


# class ROOT(object):
#     def __init__(self):
#         self.x = 10

#     def func(self, rank):
#         print("Rank=%d is able to calculate the result: %d " % (rank, np.power(self.x, 2)))

# obj= None
# if rank == 0:
#     obj = ROOT()
# obj = comm.bcast(obj,root=0)

# obj.func(rank)


