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
import configparser

config = configparser.ConfigParser()
config.read("src/config.ini")

SETTING = config["SETTING"]["setting"]

env2 = GrayscaleObservation(env)
env3 = ResizeObservation(env2,(84,84))
env4 = FrameStackObservation(env3,4)
env4.reset()
size = 500

for j in range(size):
    x = []
    for i in range(4):
        obs,_,_,_,_=env4.step(0)
        x.append(obs)
    
    np.save(f"src/data/setting{SETTING}",np.array(x)/255.0)
    env4.reset()