import gymnasium as gym
import keras
import tensorflow as tf
import time
from keras import layers
import pygame
from gymnasium.wrappers import FrameStackObservation,GrayscaleObservation,ResizeObservation,RescaleObservation

import matplotlib.pyplot as plt
import keras
import numpy as np

env = gym.make("baxter_robot:robot-arm-v1")
env = GrayscaleObservation(env)
env = ResizeObservation(env,(84,84))

# # Stack four frames
env = FrameStackObservation(env, 4)

env.reset()


class TransposeLayer(layers.Layer):
    def call(self, inputs):
        return tf.transpose(inputs, perm=[0, 2, 3, 1])

model = keras.models.load_model("src/model/modelC0.keras",{"TransposeLayer":TransposeLayer})

for i in range(4):
    obs,_,_,_,_=env.step(0)
done = False
t = 0
while t< 1000:
    obs = np.array(obs)/255.0
    state_tensor = keras.ops.convert_to_tensor(obs)
    state_tensor = keras.ops.expand_dims(state_tensor, 0)
    action_probs = model.predict(state_tensor)
            # Take best action
    action = keras.ops.argmax(action_probs[0]).numpy()
    
    obs, reward, done, _, info = env.step(action)

    
    print("reward:",reward)
    print("info:",info)
    print("Action",action)
    plt.imsave("here2.png",obs[-1],cmap="grey")
    time.sleep(0.2)
    t+=1
    