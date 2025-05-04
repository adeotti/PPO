import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torchvision.transforms import v2,Resize
from torchvision.transforms.functional import to_tensor
from PIL import Image
from gymnasium.wrappers import FrameStack
import warnings
warnings.filterwarnings("ignore")

env = gym_super_mario_bros.make('SuperMarioBros-v0',apply_api_compatibility=True,render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = FrameStack(env,5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def _transform(observation):
    _list_ = []
    for element in observation:
        tonumpy = np.array(element)
        topil = Image.fromarray(tonumpy)
        grayscale = to_tensor(v2.Grayscale(1)(topil))
        resized = Resize((150,150))(grayscale)
        _list_.append(resized)
    return torch.stack(_list_,dim=0).permute(1,0,2,3)


class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.LazyConv2d(1,(1,1))
        self.conv1 = nn.LazyConv2d(1,(3,3),stride=2)
        self.conv2 = nn.LazyConv2d(1,(3,3),stride=1)
        self.conv3 = nn.LazyConv2d(1,(3,3),stride=2)
        
        self.linear1 = nn.LazyLinear(3000)
        self.linear2 = nn.LazyLinear(1500)
        self.linear3 = nn.LazyLinear(750)
        self.linear4 = nn.LazyLinear(375)

        self.policyHead = nn.LazyLinear(7)
        self.valueHead = nn.LazyLinear(1)
        
    def forward(self,x):
        x = F.relu(self.input(x))
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(torch.flatten(x,start_dim=1))
        x = self.linear1(x)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.relu(self.linear4(x))
        policyOut = self.policyHead(x)
        valueOut = self.valueHead(x)
        return F.softmax(policyOut,-1),valueOut
    
network()(torch.rand((1,5,150,150),dtype=torch.float))
model = network()
model.load_state_dict(torch.load("./data/2_mario200",map_location=device),strict=False)


if __name__ == "__main__":
    done = True
    for step in range(5000):
        if done:
            state,_ = env.reset()
        state = _transform(state)
        dist,_ = model.forward(state)
        action = Categorical(dist).sample().item()
        state, reward, done, info,_ = env.step(action)
        print(reward)
        env.render()

    env.close()

