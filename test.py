import warnings
warnings.filterwarnings("ignore")
import torch,gym,random
from dataclasses import dataclass
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation,FrameStack,ResizeObservation
from gym.vector import SyncVectorEnv
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4,32,1,1,0)
        self.conv2 = nn.Conv2d(32,32,3,2,2)
        self.conv3 = nn.Conv2d(32,32,3,2,2)
        self.conv4 = nn.Conv2d(32,32,3,2,2)
        self.output = nn.Linear(169,80)

        self.policy_head = nn.Linear(80,7)
        self.value_head = nn.Linear(80,1)
        self.optim = torch.optim.Adam(self.parameters(),lr=1)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x,start_dim=1) # -> torch.Size([32, 169])
        x = F.relu(self.output(x))
        policy_output = self.policy_head(x)
        value_output = self.value_head(x)
        return F.softmax(policy_output,-1),value_output 



class test:
    @staticmethod
    def make_env(): 
        class CustomEnv(nn.Module):
            def __init__(self):
                pass

        x = gym_super_mario_bros.make("SuperMarioBros-v1",apply_api_compatibility=True,render_mode="human") 
        x = ResizeObservation(x,(100,100))
        x = CustomEnv(x,4) 
        x = JoypadSpace(x, SIMPLE_MOVEMENT)  
        x = GrayScaleObservation(x,True)
        x = FrameStack(x,4)
        return x

    @staticmethod
    def run(start,num_game):
        if start:
            with torch.no_grad():
                model = network()
                chk = torch.load(".\mario960",map_location="cpu")
                model.load_state_dict(chk["model_state"],strict=False)
                env = __class__.make_env()
                done = True
                epi_rewards = 0
                for _ in range(num_game):
                    if done:
                        state,_ = env.reset()
                        print(epi_rewards)
                        epi_rewards = 0
                    state = torch.from_numpy(np.array(state).copy()).squeeze().to("cpu",torch.float32).unsqueeze(0)
                    dist,_ = model.forward(state)
                    action = Categorical(dist).sample().item()
                    state, reward, done, info,_ = env.step(env.action_space.sample())
                    epi_rewards += reward
                    env.render()
                env.close()

test.run(start=False,num_game=10_000)