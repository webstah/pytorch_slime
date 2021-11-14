import numpy as np
import cv2

import argparse
import torch
from torchvision import transforms

from agent import Agents


def main(config):
    h = config.env_height
    w = config.env_width
    decay_rate = 0.8

    # initialize frame and agents
    frame = torch.zeros((w, h))
    print(frame.shape)
    agents = Agents(width=w, height=h, num_agents=config.num_agents, 
                    move_speed=0.5)

    # for blurring trails
    blur = transforms.Compose([transforms.GaussianBlur(5)])

    # initial update to frame
    frame = frame.index_put_(indices=agents.get_pos(), values=torch.ones(config.num_agents))

    for i in range(config.max_frames):
        # update agent population
        agents.update(frame)

        # update frame
        frame = frame * decay_rate
        frame = blur(torch.unsqueeze(torch.unsqueeze(frame, 0), 0))
        frame = torch.squeeze(frame)
        frame = frame.index_put_(indices=agents.get_pos(), values=torch.ones(config.num_agents))

        # display image image
        scale = 60
        w = int(frame.shape[1] * scale / 100)
        h = int(frame.shape[0] * scale / 100)
        img = frame.transpose(1, 0).cpu().detach().numpy()
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imshow("frame", resized)
        if i % 100 == 0:
            cv2.imwrite('./data/'+f'{i:06}'+'.jpg', resized*255)
        key = cv2.waitKey(1) & 0xFF

        #kill loop on key press
        if key == ord("q"):
            break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch_slime')
    parser.add_argument('--num-agents', dest='num_agents',
                        default=200000, type=int,
                        help='Number of agents in environment.')
    parser.add_argument('--env-height', dest='env_height',
                        default=1000, type=int,
                        help='Height of the environment in pixels.')
    parser.add_argument('--env-width', dest='env_width',
                        default=1000, type=int,
                        help='Height of the environment in pixels.')
    parser.add_argument('--max-frames', dest='max_frames',
                        default=100000, type=int,
                        help='Max number of frames.')

    config = parser.parse_args()

    assert config.num_agents < (config.env_height * config.env_width), \
        'More agents than there are number of pixels. Reduce number of agents.'

    main(config)