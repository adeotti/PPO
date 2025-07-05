import gymnasium as gym
import torch,sys
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class network(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.LazyLinear(32)
        self.l2 = nn.LazyLinear(32)
        self.l3 = nn.LazyLinear(32)
        self.policy = nn.LazyLinear(2)
        self.value = nn.LazyLinear(1)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        policy = self.policy(x)
        value = self.value(x)        
        return F.softmax(policy,-1),value
    
model = network().to("cpu")
model(torch.rand((5,4),dtype=torch.float32,device="cpu"))
chk = torch.load("CartPole\CartPole.pth",map_location="cpu")
model.load_state_dict(chk["model_state"],strict=False)

env = gym.make("CartPole-v1", render_mode="human")

obs, _ = env.reset()
total_reward = 0

for step in range(10000):
    probs,_ = model(torch.from_numpy(obs).to(torch.float32))
    action = Categorical(probs).sample().tolist()
    #action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    if done or truncated:
        print(f"END with total reward: {total_reward}")
        obs, _ = env.reset()
        total_reward = 0

env.close()


