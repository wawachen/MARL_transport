from pyrep import PyRep
import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection,displaySystem

from os import path
from os.path import dirname, join, abspath
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

from pyrep.objects.vision_sensor import VisionSensor
import random
import math
import torch

class Magnet_env:
    def __init__(self,env_name):
        self.pr = PyRep()
        self.pr.launch(env_name, headless=False)
        self.pr.start()
        self.robot = Shape('Magbot')

        self.agents = []
        for i in range(12):
            for j in range(12):
                self.agents.append(Cylinder(mag=(0,0,500), dim=(100,200),pos=(((6-i)*100-50),((6-j)*100-50),200)))
        
        self.magnet = Collection(self.agents)
        # self.Bs = self.magnet.getB(POS)

    def step(self):
        
        self.pr.step()
        
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


# x1 = Box(mag=(0,0,600), dim=(3,3,3), pos=(-4,0,3))
# x2 = Cylinder(mag=(0,0,500), dim=(3,5))

# c = Collection(x1,x2)

try:
    env_name = join(dirname(abspath(__file__)), 'magnet_field.ttt')
    env = Magnet_env(env_name)
    while 1:
        env.step()
finally:
    env.shutdown()

# xs = np.linspace(-2500,2500,100)
# zs = np.linspace(0,1000,80)

# POS = np.array([(x,0,z) for z in zs for x in xs])
# Bs = env.magnet.getB(POS).reshape(80,100,3)  

# fig = plt.figure(figsize=(9,5))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122) 

# displaySystem(env.magnet, subplotAx=ax1, suppress=True)
# X,Z = np.meshgrid(xs,zs)
# U,V = Bs[:,:,0], Bs[:,:,2]
# ax2.streamplot(X, Z, U, V, color=np.log(U**2+V**2))

# plt.show()