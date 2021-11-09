import torch
from torch import nn
import argparse

def main(config):

    return None






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='graph_ae')
    parser.add_argument('--gpus', default=1, type=int,
                        help='Number of GPUs to use.')
    parser.add_argument('--num-agents', dest='num_workers',
                        default=1000, type=int,
                        help='Number of agents in environment.')
    parser.add_argument('--env-height', dest='env_height',
                        default=720, type=int,
                        help='Height of the environment in pixels.')
    parser.add_argument('--env-width', dest='env_width',
                        default=128, type=int,
                        help='Height of the environment in pixels.')
    parser.add_argument('--max-epochs', dest='max_epochs',
                        default=1000, type=int,
                        help='Max number of epochs.')

    config = parser.parse_args()

    assert config.num_agents < (config.env_height * config.env_width), \
        'More agents than there are number of pixels. Reduce number of agents.'

    main(config)