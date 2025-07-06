import torch,gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation,FrameStack,ResizeObservation
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = "cpu"

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.LazyConv2d(32,1,1,0)
        self.conv2 = nn.LazyConv2d(32,3,2,2)
        self.conv3 = nn.LazyConv2d(32,3,2,2)
        self.conv4 = nn.LazyConv2d(32,3,2,2)
        self.output = nn.LazyLinear(80)

        self.policy_head = nn.LazyLinear(7)
        self.value_head = nn.LazyLinear(1)
        #self.optim = torch.optim.Adam(self.parameters(),lr=configs.lr)
        
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

def init_weights(layer):
    if isinstance(layer,(nn.Conv2d,nn.Linear)):
        nn.init.orthogonal_(layer.weight)
        nn.init.constant_(layer.bias,0.0)
     
model = network().to(device)
model.forward(
    torch.rand(
        (1,4,100,100),dtype=torch.float32,device=device
    )
)
model.apply(init_weights)
model = nn.DataParallel(model)

class test:
    @staticmethod
    def make_env(): 
        class CustomEnv(gym.Wrapper): 
            def __init__(self,env,skip):
                super().__init__(env)
                self.skip = skip

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
                chk = torch.load("Mario\mario280",map_location="cpu")
                model.load_state_dict(chk["model_state"],strict=False)
                env = __class__.make_env()
                done = True
                epi_rewards = 0
                for _ in range(num_game):
                    if done:
                        state,_ = env.reset()
                        print(epi_rewards)
                        epi_rewards = 0
                        import sys
                    #state = torch.from_numpy(np.array(state)).permute(-1,0,1,2).to(torch.float32) / 255.
                    state = torch.from_numpy(np.array(state)).permute(-1,0,1,2).to(device,torch.float32) / 255.
                    dist,_ = model(state)
                    action = Categorical(dist).sample().item()
                    state, reward, done, info,_ = env.step(action)
                    epi_rewards += reward
                    env.render()
                env.close()


if __name__ == "__main__":
    test.run(start=True,num_game=10_000)
     