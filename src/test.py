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
import configparser

config = configparser.ConfigParser()
config.read("src/config.ini")

env = gym.make("baxter_robot:robot-arm-v1")
env = GrayscaleObservation(env)
env = ResizeObservation(env,(84,84))
# # Stack four frames
env = FrameStackObservation(env, 4)
env.reset()


class TransposeLayer(layers.Layer):
    def call(self, inputs):
        return tf.transpose(inputs, perm=[0, 2, 3, 1])

TEST_SETTING = config["SETTING"]["setting"]
model_type = ["A","B","C","D","E"]
iter_list = ["10000","20000","40000","60000","80000"]
for model,iter in zip(model_type,iter_list):
    model_name = "model" + model + iter
    model = keras.models.load_model(f"src/model/{model_name}.keras",{"TransposeLayer":TransposeLayer})

    success_radius = 15
    success = 0    
    test_steps = 200
    for i in range(test_steps): 
        t = 0
        env.reset() 
        for i in range(4):
            obs,_,_,_,_=env.step(0)  
        while t< 1000:
            obs = np.array(obs)/255.0
            state_tensor = keras.ops.convert_to_tensor(obs)
            state_tensor = keras.ops.expand_dims(state_tensor, 0)
            action_probs = model.predict(state_tensor)
                    # Take best action
            action = keras.ops.argmax(action_probs[0]).numpy()    
            obs, reward, done, _, info = env.step(action)
            if info["distance_error"]  < success_radius:
                success+=1
                break
            t +=1
                    
    with open("src/data/report.txt",mode = "a") as file:
        file.write(f"Success rate for model {model_name}  on setting {TEST_SETTING} : "+str(success/test_steps * 100 )+"%.\n")
        
    
    