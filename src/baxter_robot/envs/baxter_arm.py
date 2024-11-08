import pygame
import math
import numpy as np

WHITE = [229, 229, 229]
BLACK = [38,51,51]
GREY = [63,76,76]
ORANGE = [178,63,51]
BLUE = [15,10,252]

class Link():
    def __init__(self,length,width,color):
        self.link_length = length
        self.link_width = width
        self.link_color = color

class BaxterArm():
    def __init__(self,link = [27,34.8,36.5,22.6],initial_pose = [50,160]):
        self.width = 160
        self.height = 320
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill(WHITE)
        self.common_link_width = 14.7/251.1 * self.height
        self.eef_width = 13/251.1*self.height
        self.initial_pose = initial_pose               
        self.ground_link = Link(length=link[0],width=self.common_link_width,color=GREY)
        self.link_s1e1 = Link(length=link[1],width=self.common_link_width,color=ORANGE)
        self.link_e1w1 = Link(length=link[2],width=self.common_link_width,color=ORANGE)
        self.eef = Link(length=link[3],width=self.eef_width,color=GREY)
        
        self.s1_diameter = self.link_s1e1.link_width * 1.1
        self.e1_diameter = self.link_s1e1.link_width * 1.3
        self.w1_diameter = self.e1_diameter
        
        self.ground_s1_dist = 0.45 * self.link_s1e1.link_width
        self.e1_offset = 0.532 * self.link_s1e1.link_width
        self.w1_offset = 0
        self.target = self.get_target() 
        
        
    def render(self):
        pygame.display.update()
    
    def draw_joint(self,point,diameter,color):
        pygame.draw.circle(self.screen,color,point,diameter/2)
        return list(point)
    
    def draw_link(self,point,theta,link,angle_mode="radians"):        
        link_width,link_length,link_color = link.link_width,link.link_length,link.link_color
        x, y = point
        half_width = link_width/2
        if angle_mode == "degrees":
            theta = math.radians(theta)
        point1 = point
        point2 = (x + half_width*math.sin(theta) , y + half_width*math.cos(theta))
        point3 = (x + half_width*math.sin(theta) + link_length*math.cos(theta) , y + half_width*math.cos(theta)-link_length*math.sin(theta))
        point4 = (x - half_width*math.sin(theta) + link_length*math.cos(theta) , y - half_width*math.cos(theta)-link_length*math.sin(theta))
        point5 = (x - half_width*math.sin(theta) , y - half_width*math.cos(theta))
        pygame.draw.polygon(self.screen,link_color,[point,point1,point2,point3,point4,point5])
        link_end = [x + link_length*math.cos(theta),y-link_length*math.sin(theta)]
        return link_end
    
    def draw_robot(self,theta):
        self.screen.fill(WHITE)
        self.draw_target()
        theta1 , theta2, theta3 = [sum(theta[:i+1]) for i in range(len(theta))]
        self.draw_joint(self.initial_pose,self.ground_link.link_width,self.ground_link.link_color)
        point = self.draw_link(self.initial_pose,math.pi/2,self.ground_link)
        point = self.draw_joint(point,self.ground_link.link_width,self.ground_link.link_color)
        point[0] = point[0] + self.ground_s1_dist    
        #draw link s1e1
        point = self.draw_joint(point,self.s1_diameter,self.link_s1e1.link_color)
        point = self.draw_link(point,theta1,self.link_s1e1)
        point = self.draw_joint(point,self.s1_diameter,self.link_s1e1.link_color)    
        #calculate e1    
        point = (point[0] + self.e1_offset*math.sin(theta1), point[1] + self.e1_offset*math.cos(theta1))
        e1 = point    
        #draw link e1w1
        point = self.draw_link(point,theta2,self.link_e1w1)
        self.draw_joint(e1,self.e1_diameter,BLACK)    
        #draw end effector
        point = (point[0] + self.w1_offset*math.sin(theta2), point[1] + self.w1_offset*math.cos(theta2))
        w1 = point
        point = self.draw_link(point,theta3,self.eef)
        self.draw_joint(w1,self.w1_diameter,BLACK)
        self.end_effecter_pos = point
        
        return self.end_effecter_pos
    
    def draw_target(self):
        pygame.draw.rect(self.screen, BLUE, pygame.Rect(self.target[0] - 15, self.target[1] - 15, 30, 30))
        #return self.draw_joint(self.target,30,BLUE)
    
    def get_target(self):
        # Generate random numbers
        ground_joint_x = self.initial_pose[0]
        ground_joint_y = self.initial_pose[1] - self.ground_link.link_length
        link_reach = self.link_e1w1.link_length + self.link_s1e1.link_length + self.eef.link_length
        x = np.random.randint(min(0,abs(ground_joint_x - link_reach)),min(ground_joint_x + link_reach,self.width))
        y_lim = np.sqrt(abs(((x-ground_joint_x)**2 - link_reach**2)))
        y = np.random.randint(min(0,abs(ground_joint_y - link_reach)),ground_joint_y + y_lim)

        # Combine into a shape (1, 2) array or tuple
        target = (x, y)
        return target
        
        
    def get_screen(self):
        return np.transpose(pygame.surfarray.array3d(self.screen),(1,0,2))