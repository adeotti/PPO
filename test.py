import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from gym.wrappers import GrayScaleObservation,ResizeObservation
import warnings
warnings.filterwarnings("ignore")

num_frames = 4
image_shape = (90,90)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env():
    x = gym_super_mario_bros.make('SuperMarioBros-v0',apply_api_compatibility=True,render_mode="human")
    x = JoypadSpace(x, SIMPLE_MOVEMENT)
    x = ResizeObservation(x,image_shape)
    x = GrayScaleObservation(env=x,keep_dim=True)
    return x

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
   
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x,start_dim=1)  
        x = F.relu(self.output(x))
        policy_output = self.policy_head(x)
        value_output = self.value_head(x)
        return F.softmax(policy_output,-1),value_output

model = network().to(device)
model.forward(torch.rand((1,1,90,90),device=device,dtype=torch.float))
chk = torch.load("./mario20.txt",map_location=device)
model.load_state_dict(chk["model_state"],strict=False)

if __name__ == "__main__":
    env = make_env()
    done = True
    for step in range(20000):
        if done:
            state,_ = env.reset()
        state = torch.from_numpy(state).permute(-1,0,1).to(torch.float32).unsqueeze(0)
        dist,_ = model.forward(state)
        action = Categorical(dist).sample().item()
        state, reward, done, info,_ = env.step(action)
    env.close()

