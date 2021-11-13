import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import one_hot_categorical


class AgentUpdate(nn.Module):
    def __init__(self, width, height, move_speed, sensor_offset=0.6, p_t=0.005):
        super(AgentUpdate, self,).__init__()
        self.sensor_offset = sensor_offset
        self.p_t = p_t
        self.width = width
        self.height = height
        self.move_speed = move_speed

    def clip(self, x, min, max):
        return torch.max(min, torch.min(x, max))

    def forward(self, x, y, theta, frame):
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        #### get next positions ####
        # get sensor positions
        sensor_x_l = x + torch.cos(theta-self.sensor_offset) * self.move_speed
        sensor_y_l = y + torch.sin(theta-self.sensor_offset) * self.move_speed
        sensor_x_r = x + torch.cos(theta+self.sensor_offset) * self.move_speed
        sensor_y_r = y + torch.sin(theta+self.sensor_offset) * self.move_speed
        sensor_x_c = x + torch.cos(theta) * self.move_speed
        sensor_y_c = y + torch.sin(theta) * self.move_speed

        # correct results that are out of bounds
        sensor_x_l = self.clip(sensor_x_l, zeros, ones*(self.width-1))
        sensor_y_l = self.clip(sensor_y_l, zeros, ones*(self.height-1))
        sensor_x_r = self.clip(sensor_x_r, zeros, ones*(self.width-1))
        sensor_y_r = self.clip(sensor_y_r, zeros, ones*(self.height-1))
        sensor_x_c = self.clip(sensor_x_c, zeros, ones*(self.width-1))
        sensor_y_c = self.clip(sensor_y_c, zeros, ones*(self.height-1))

        # get detections from sensors
        det_l = frame[sensor_x_l.type(torch.LongTensor), sensor_y_l.type(torch.LongTensor)]
        det_r = frame[sensor_x_r.type(torch.LongTensor), sensor_y_r.type(torch.LongTensor)]
        det_c = frame[sensor_x_c.type(torch.LongTensor), sensor_y_c.type(torch.LongTensor)]

        l_x = torch.cos(theta-self.sensor_offset)
        l_y = torch.sin(theta-self.sensor_offset)
        r_x = torch.cos(theta+self.sensor_offset)
        r_y = torch.sin(theta+self.sensor_offset)
        c_x = torch.cos(theta)
        c_y = torch.sin(theta)

        # ! we need to scale our logits between 0 and 1 along dim 1
        logits = F.softmax(torch.stack([det_l, det_r, det_c], dim=0).transpose(1,0), dim=1)
        choices_x = torch.stack([l_x, r_x, c_x] ,dim=0).transpose(1,0)
        choices_y = torch.stack([l_y, r_y, c_y], dim=0).transpose(1,0)
        sample = one_hot_categorical.OneHotCategorical(logits=logits, validate_args=False).sample()

        sampled_x = torch.sum(sample * choices_x, dim=1)
        sampled_y = torch.sum(sample * choices_y, dim=1)

        # randomy change the angle of each agent with probability p_t
        # theta_rand = torch.rand_like(x.type(torch.FloatTensor)) * 2 * 3.141592
        # prob = torch.rand_like(x.type(torch.FloatTensor))
        # theta = torch.where(prob <= self.p_t, theta_rand, theta)

        # calculate new positions
        x = x + sampled_x * self.move_speed
        y = y + sampled_y * self.move_speed
        
        #### correct coordinates that are outside of the frame ####
        # twos tensors of size x and y filled with ones and zeros are needed
        
        theta_rand = torch.rand_like(x) * 2 * 3.141592
        # check x and y coordinates
        x_clip = torch.where(x >= self.width, 
                            torch.max(zeros, torch.min(x, ones*(self.width-1))), x)
        x_clip = torch.where(x <= 0, 
                            torch.max(zeros, torch.min(x, ones*(self.width-1))), x_clip)
        y_clip = torch.where(y >= self.height, 
                            torch.max(zeros, torch.min(y, ones*(self.height-1))), y)
        y_clip = torch.where(y <= 0, 
                            torch.max(zeros, torch.min(y, ones*(self.height-1))), y_clip)

        # give new random angles to agents that hit the boundary
        theta_clip = torch.where(x >= self.width, ones, zeros)
        theta_clip = theta_clip + torch.where(x <= 0, ones, zeros)
        theta_clip = theta_clip + torch.where(y >= self.height, ones, zeros)
        theta_clip = theta_clip + torch.where(y <= 0, ones, zeros)

        # recombine new and old 
        theta_clip = theta_clip*theta_rand + (torch.abs(theta_clip - 1))*theta

        return x_clip, y_clip, theta_clip

class Agents:
    def __init__(self, width, height, num_agents=64, move_speed=1):
        self.move_speed = move_speed
        self.x = torch.randint(low=width//2-50, high=width//2+50, size=(num_agents,))
        self.y = torch.randint(low=height//2-50, high=height//2+50, size=(num_agents,))
        self.theta = torch.rand(size=(num_agents,)) * 2 * 3.141592
        # cen_x = width//2
        # cen_y = height//2
        # top = cen_y - self.y
        # bot = cen_x - self.x
        # self.theta = torch.atan(top/bot)
        # print(self.theta)
        self.get_update = AgentUpdate(width, height, move_speed)

    def update(self, frame):
        self.x, self.y, self.theta = self.get_update(self.x, self.y, self.theta, frame)

    def get_pos(self):
        coord = torch.stack((self.x, self.y), dim=0)

        return list(coord.type(torch.LongTensor))

if __name__ == '__main__':
    agents = Agents(height=1080, width=1920, num_agents=256, move_speed=5)
    for i in range(1000):
        # print(agents.get_pos())
        agents.update()
        print(" ")
