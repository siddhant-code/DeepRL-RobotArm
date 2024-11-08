import gymnasium as gym
import keras
import tensorflow as tf
import numpy as np
import time
from keras import layers
import pygame
from gymnasium.wrappers import FrameStackObservation,GrayscaleObservation,ResizeObservation
env = gym.make("baxter_robot:robot-arm-v1")
import matplotlib.pyplot as plt
import keras

env2 = GrayscaleObservation(env)
env3 = ResizeObservation(env2,(84,84))
env4 = FrameStackObservation(env3,4)
env4.reset()
size = 500

for j in range(500):
    x = []
    for i in range(4):
        obs,_,_,_,_=env4.step(0)
        x.append(obs)
    
    np.save("src/data/settingB",np.array(x)/255.0)
    env4.reset()