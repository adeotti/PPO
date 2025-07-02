import torch,gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation,FrameStack,ResizeObservation
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.LazyConv2d(32,1,1,0)
        self.conv2 = nn.LazyConv2d(32,3,2,2)
        self.conv3 = nn.LazyConv2d(32,3,2,2)
        self.conv4 = nn.LazyConv2d(32,3,2,2)
        self.output = nn.LazyLinear(80)
        self.policy_head = nn.LazyLinear(7)
       
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x,start_dim=1) # -> torch.Size([32, 169])
        x = F.relu(self.output(x))
        policy_output = self.policy_head(x)
        return F.softmax(policy_output,-1) 

m = network()
m.forward(torch.rand(1,4,100,100))

class test:
    @staticmethod
    def make_env(): 
        class CustomEnv(gym.Wrapper): 
            def __init__(self,env,skip):
                super().__init__(env)
                self.skip = skip
                self.score = 0
                
            def step(self, action):
                total_reward = 0  
                for _ in range(self.skip):
                    obs,reward,done,truncared,info = self.env.step(action)
                    total_reward += reward 
                    if done:
                        self.reset()
                        return obs,(total_reward/10.),done,truncared,info
                return obs,(total_reward/10.),done,truncared,info

            def reset(self, **kwargs):
                self.score = 0
                obs, info = self.env.reset(**kwargs)
                return obs,info

        x = gym_super_mario_bros.make("SuperMarioBros-v0", apply_api_compatibility=True,render_mode="human")
        x = JoypadSpace(x, SIMPLE_MOVEMENT)
        x = ResizeObservation(x,(100,100))
        x = CustomEnv(x,4)
        x = GrayScaleObservation(x, keep_dim=True)
        x = FrameStack(x,4) 
        return x

    @staticmethod
    def run(start,num_game):
        if start:
            with torch.no_grad():
                model = network()
                chk = torch.load(".\\9_mario90",map_location="cpu")
                model.load_state_dict(chk["model_state"],strict=False)
                env = __class__.make_env()
                done = True
                epi_rewards = 0
                for _ in range(num_game):
                    if done:
                        state,_ = env.reset()
                        print(epi_rewards)
                        epi_rewards = 0
                    state = torch.from_numpy(np.array(state)).permute(-1,0,1,2).to(torch.float32) / 255.
                    dist = model.forward(state)
                    action = Categorical(dist).sample().item()
                    state, reward, done, info,_ = env.step(action)
                    epi_rewards += reward
                    env.render()
                env.close()


if __name__ == "__main__":
    test.run(start=True,num_game=10_000)
     