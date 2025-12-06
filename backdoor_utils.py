from __future__ import print_function

import torch
import random

def add_pattern(pattern):

    trigger = None

    if pattern == "sqr" :
        trigger = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], 
                   [0, 4, 0], [0, 4, 1], [0, 4, 2], [0, 4, 3], [0, 4, 4], 
                   [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 1, 4], [0, 2, 4], 
                   [0, 3, 4], ] 
    elif pattern == "hsh" :
        trigger = [[0, 0, 1], [0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1], 
                   [0, 0, 3], [0, 1, 3], [0, 2, 3], [0, 3, 3], [0, 4, 3], 
                   [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4], 
                   [0, 3, 0], [0, 3, 1], [0, 3, 2], [0, 3, 3], [0, 3, 4], ]
    elif pattern == "crs" :
        trigger = [[0, 4, 4], [0, 3, 3], [0, 2, 2], [0, 1, 1], [0, 0, 0], 
                   [0, 2, 2], [0, 1, 3], [0, 0, 4], [0, 3, 1], [0, 4, 0], ]
    elif pattern == "pls" :
        trigger = [[0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 2, 3], [0, 2, 4], 
                   [0, 0, 2], [0, 1, 2], [0, 2, 2], [0, 3, 2], [0, 4, 2], ]
    elif pattern == "eql" :
        trigger = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], 
                   [0, 4, 0], [0, 4, 1], [0, 4, 2], [0, 4, 3], [0, 4, 4], ]
    elif pattern == "prl" :
        trigger = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], 
                   [0, 0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4], [0, 4, 4], ]
    if trigger is None:
        raise ValueError(f"Unknown pattern: {pattern}")
    return trigger 

class Backdoor_Utils():

    def __init__(self, backdoor_label = 10, backdoor_pattern = 'pls', backdoor_fraction = 0.5):
        self.backdoor_label = backdoor_label
        self.backdoor_pattern = backdoor_pattern
        self.backdoor_fraction = backdoor_fraction
        #self.trigger_position = [[0, 0, 0], [0, 0, 1], [0, 0, 2],   [0, 0, 4], [0, 0, 5], [0, 0, 6],\
                                 
                                # [0, 2, 0], [0, 2, 1], [0, 2, 2],   [0, 2, 4], [0, 2, 5], [0, 2, 6], ]
        #self.trigger_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]

    def get_poison_batch(self, data, targets, evaluation=False):
        #         poison_count = 0
        new_data = torch.empty(data.shape)
        new_targets = torch.empty(targets.shape)

        for index in range(0, len(data)):
            if evaluation:  # will poison all batch data when testing
                new_targets[index] = self.backdoor_label
                new_data[index] = self.add_backdoor_pixels(data[index],self.backdoor_pattern)
            #                 poison_count += 1

            else:  # will poison only a fraction of data when training
                if torch.rand(1) < self.backdoor_fraction:
                    new_targets[index] = self.backdoor_label
                    new_data[index] = self.add_backdoor_pixels(data[index], self.backdoor_pattern)
                #                     poison_count += 1
                else:
                    new_data[index] = data[index]
                    new_targets[index] = targets[index]

        new_targets = new_targets.long()
        if evaluation:
            new_data.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_data, new_targets

    def add_backdoor_pixels(self, item, pattern):

        pos = add_pattern(pattern)
        for p in pos:
                item[p[0]][p[1]][p[2]] = 1
        return item