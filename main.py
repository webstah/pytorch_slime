import argparse
import torch
from torchvision import transforms

from agent import Agents


def main(config):
    h = config.env_height
    w = config.env_width
    decay_rate = 0.9

    # initialize frame and agents
    frame = torch.zeros((h, w))
    agents = Agents(config.num_agents)

    # for blurring trails
    blur = transforms.Compose([transforms.GaussianBlur()])

    # initial update to frame
    frame = frame.index_put_(agents.get_pos(), torch.ones_like(frame))

    for i in range(config.max_frames):
        # update agent population
        agents.update(frame)

        # update frame
        frame = frame * decay_rate
        frame = blur(torch.unsqueeze(torch.unsqueeze(frame, 0), 0))
        frame = torch.squeeze(frame)

        frame = frame.index_put_(agents.get_pos(), torch.ones_like(frame))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch_slime')
    parser.add_argument('--num-agents', dest='num_agents',
                        default=1000, type=int,
                        help='Number of agents in environment.')
    parser.add_argument('--env-height', dest='env_height',
                        default=720, type=int,
                        help='Height of the environment in pixels.')
    parser.add_argument('--env-width', dest='env_width',
                        default=128, type=int,
                        help='Height of the environment in pixels.')
    parser.add_argument('--max-frames', dest='max_frames',
                        default=1000, type=int,
                        help='Max number of frames.')

    config = parser.parse_args()

    assert config.num_agents < (config.env_height * config.env_width), \
        'More agents than there are number of pixels. Reduce number of agents.'

    main(config)