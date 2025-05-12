import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

env = gym_super_mario_bros.make('SuperMarioBros-v0',apply_api_compatibility=True,render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env=env,keep_dim=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.LazyConv2d(32,(3,3),stride=2,padding=1)
        self.conv2 = nn.LazyConv2d(32,(3,3),stride=2,padding=1)
        self.conv3 = nn.LazyConv2d(64,(3,3),stride=2,padding=1)
        self.conv4 = nn.LazyConv2d(64,(3,3),stride=2,padding=1)
        self.conv5 = nn.LazyConv2d(64,(3,3),stride=2,padding=1)
        self.conv6 = nn.LazyConv2d(64,(3,3),stride=2,padding=0)
     
        self.policy_head = nn.LazyLinear(7)
        self.value_head = nn.LazyLinear(1)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = torch.flatten(x,start_dim=1)  
        policy_output = self.policy_head(x)
        value_output = self.value_head(x)
        return F.softmax(policy_output,-1),value_output

model = network()
model.forward(torch.rand((1,5,240,256),dtype=torch.float))
chck = torch.load("./",map_location=device)
model.load_state_dict(chck["model_state"],strict=False)

if __name__ == "__main__":
    done = True
    for step in range(5000):
        if done:
            state,_ = env.reset()
        state = None # TODO transform
        dist,_ = model.forward(state)
        action = Categorical(dist).sample().item()
        state, reward, done, info,_ = env.step(action)
        print(action)
        
    env.close()

