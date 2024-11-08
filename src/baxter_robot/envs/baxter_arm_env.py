import gymnasium as gym
import math
from gymnasium import spaces
from itertools import product
from .baxter_arm import BaxterArm
import numpy as np
from scipy.spatial.distance import euclidean
from collections import deque

class BaxterEnv(gym.Env):
    def __init__(self):
        self.theta = [0,0,0]
        self.arm = BaxterArm()
        self.add_noise = True
        self.noise = np.random.uniform(-0.1, 0.1, size=(320,160,3)) * 255.0
        self.previous_error = -math.inf
        # self.action = {}
        # for i,j in enumerate (product([0.02,0,-0.02],repeat=3)):
        #     self.action[i] = np.array(list(j))
        self.action = {0: np.array([0,0,0]),
                       1: np.array([0.02,0,0]),
                       2: np.array([-0.02,0,0]),
                       3: np.array([0,0.02,0]),
                       4: np.array([0,-0.02,0]),
                       5: np.array([0,0,0.02]),
                       6: np.array([0,0,-0.02])}
        self.action_space = spaces.Discrete(len(self.action))
        self.observation_space = spaces.Box(0, 255, shape = [320,160,3] , dtype=np.uint8)
    def step(self,action):
        self.theta = self.action[action] + self.theta
        end_effector = self.arm.draw_robot(self.theta) 
        distance_error = euclidean(self.arm.target, end_effector)

        reward = 0
        dis_change = distance_error - self.previous_error
        if distance_error <=0.01:
            reward = 3
        elif dis_change >= 0:
            reward = -1
        elif dis_change < 0:
            reward = 1
        else:
            reward = 0      
        # if distance_error >= self.previous_error:
        #     reward = -1
        # epsilon = 10
        # if (distance_error > -epsilon and distance_error < epsilon):
        #     reward = 1
        self.previous_error = distance_error
        self.current_score.append(reward)
        if sum(self.current_score) < -1 or distance_error == 0:
            done = True
        else:
            done = False
        observation = self.arm.get_screen()
        if self.add_noise:
            observation = (observation + np.random.uniform(-0.1, 0.1, size=(320,160,3)) * 255.0)          
        info = {
            
            'distance_error': distance_error,
            'target_position': self.arm.target,
            'current_position': end_effector
        }       
        return observation,reward,done,None,info
        
        
    def render(self):
        self.arm.render()    
    
    def reset(self,theta = [0,0,0],**kwargs):
        #self.arm = BaxterArm()
        self.arm.target = self.arm.get_target()
        self.current_score = deque([0,0,0],maxlen=3)
        end_effector = self.arm.draw_robot(theta)
        distance_error = euclidean(self.arm.target, end_effector)
        observation = self.arm.get_screen()
        info = {
            
            'distance_error': distance_error,
            'target_position': self.arm.target,
            'current_position': end_effector
        } 
        return observation,info
    
    def get_action_meanings(self):
        return ["NOOP"]
    
    