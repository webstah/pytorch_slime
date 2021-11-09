import torch
from torch import nn


class Agents:
    def __init__(self, height, width, num_agents=300):
        self.x = torch.randint(low=0, high=width, size=(num_agents,))
        self.y = torch.randint(low=0, high=height, size=(num_agents,))
        self.theta = torch.rand(size=(num_agents,))

    def update(self, frame):

        # TODO: get 3 values in front of agent from frame 
        # tensor_L = ???
        # tensor_C = ???
        # tensor_R = ???

        # TODO: update self.agents_x, self.agents_y, self.agents_theta

        return None


    def get_pos(self):
        coord = torch.stack((self.x, self.y), dim=0)
        coord = torch.transpose(coord, 0, 1)

        return tuple(coord.type(torch.LongTensor))




if __name__ == '__main__':
    agents = Agents(height=100, width=100, num_agents=32)
    print(agents.x, agents.y)
    print(agents.get_pos())