from os import path
from os.path import dirname, join, abspath
from pickle import load
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

sns.set(font_scale=1.5)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

env_name = join(dirname(abspath(__file__)), 'RL_QA{}.ttt'.format(rank))

pr = PyRep()
pr.launch(env_name, headless=False)
pr.start()

vs_floor = VisionSensor('Video_recorder') 

try:
    kk = 0
    while 1:
        depth_fl = vs_floor.capture_depth()
        img_s = np.uint8(depth_fl * 256.)
        cv2.imwrite("image_{}.png".format(rank), img_s)

        # print(depth_fl.min(),depth_fl.max())
        with open('depth_{}.npy'.format(rank), 'wb') as f:
            np.save(f, depth_fl)

        if kk>150:
            # print(depth_fl.min(),depth_fl.max())
            # ground_depth = depth_fl.min()
            # load_depth = depth_fl[np.where(depth_fl>ground_depth)]
            # print(load_depth.shape)

            ax = sns.heatmap(depth_fl)
            
            # plt.plot(load_depth)
            plt.show()
            
        pr.step()
        kk+= 1
except KeyboardInterrupt:
    pr.stop()
    pr.shutdown()

