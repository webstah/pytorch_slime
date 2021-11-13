import torch
from torch import nn


class AgentUpdate(nn.Module):
    def __init__(self, height, width, move_speed, p_t=0.01):
        super(AgentUpdate, self,).__init__()
        self.p_t = p_t
        self.width = width
        self.height = height
        self.move_speed = move_speed

    def forward(self, x, y, theta, frame):
        #### get next positions ####
        x = x + torch.cos(theta) * self.move_speed
        y = y + torch.sin(theta) * self.move_speed
        # randomy change the angle of each agent with probability p_t
        theta_rand = torch.rand_like(x) * 2 * 3.141592
        prob = torch.rand_like(x)
        theta = torch.where(prob <= self.p_t, theta_rand, theta)


        #### correct coordinates that are outside of the frame ####
        # twos tensors of size x and y filled with ones and zeros are needed
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        theta_rand = torch.rand_like(x) * 2 * 3.141592
        # check x and y coordinates
        x_clip = torch.where(x >= self.width, 
                            torch.max(zeros, torch.min(x, ones*self.width)), x)
        x_clip = torch.where(x <= 0, 
                            torch.max(zeros, torch.min(x, ones*self.width)), x_clip)
        y_clip = torch.where(y >= self.height, 
                            torch.max(zeros, torch.min(y, ones*self.height)), y)
        y_clip = torch.where(y <= 0, 
                            torch.max(zeros, torch.min(y, ones*self.height)), y_clip)

        # give new random angles to agents that hit the boundary
        theta_clip = torch.where(x >= self.width, ones, zeros)
        theta_clip = theta_clip + torch.where(x <= 0, ones, zeros)
        theta_clip = theta_clip + torch.where(y >= self.height, ones, zeros)
        theta_clip = theta_clip + torch.where(y <= 0, ones, zeros)

        # recombine new and old 
        theta_clip = theta_clip*theta_rand + (torch.abs(theta_clip - 1))*theta

        return x_clip, y_clip, theta_clip

class Agents:
    def __init__(self, height, width, num_agents=64, move_speed=1):
        self.move_speed = move_speed
        self.x = torch.randint(low=0, high=width, size=(num_agents,))
        self.y = torch.randint(low=0, high=height, size=(num_agents,))
        self.theta = torch.rand(size=(num_agents,)) * 2 * 3.141592
        self.get_update = AgentUpdate(height, width, move_speed)

    def update(self, frame):
        self.x, self.y, self.theta = self.get_update(self.x, self.y, self.theta, frame)

    def get_pos(self):
        coord = torch.stack((self.x, self.y), dim=0)
        coord = torch.transpose(coord, 0, 1)

        return tuple(coord.type(torch.LongTensor))

if __name__ == '__main__':
    agents = Agents(height=1080, width=1920, num_agents=256, move_speed=25)
    for i in range(1000):
        # print(agents.get_pos())
        agents.update()
        print(" ")
