from multiprocessing import Process
from pyrep import PyRep
from os.path import dirname, join, abspath


PROCESSES = 3

def run():
  env_name = join(dirname(abspath(__file__)), 'scene_turtlebot_navigation.ttt')
  pr = PyRep()
  pr.launch(env_name, headless=True)
  pr.start()
  # Do stuff...
  pr.stop()
  pr.shutdown()

processes = [Process(target=run, args=()) for i in range(PROCESSES)]
[p.start() for p in processes]
[p.join() for p in processes]