import torch
from torch import nn


class Agents:
    def __init__(self, height, width, num_agents=300):
        self.agents_x = torch.randint(low=0, high=width, size=num_agents)
        self.agents_y = torch.randint(low=0, high=height, size=num_agents)
        self.agents_theta = torch.rand(size=num_agents)

    def update(self, frame):