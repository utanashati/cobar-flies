"""light_times.py
BEFORE RUNNING THIS: run prepare_test.py

1. For each iteration of each experiment, determine the frame when the light
   is turned on, store in `imgs_dicts["light_times"]`.
2. For each iteration, center the frames around its `light_time`, compute
   minimum and maximum, store in `imgs_dicts["centered_{min, max}"]`.
"""

import json
import torch
import glob

data_folder = "data/orig/"

with open(data_folder + "imgs_dicts.json", 'r') as f:
    imgs_dicts = json.load(f)

imgs_dicts["light_times"] = []
imgs_dicts["centered_min"] = []
imgs_dicts["centered_max"] = []

# For each gene
for i in range(7):
    imgs_dicts["light_times"].append([])
    imgs_dicts["centered_min"].append([])
    imgs_dicts["centered_max"].append([])

    # For each iteration
    for j in range(10):
        fname = next(glob.iglob(data_folder +
                                "{:02}*/*{:02}/"
                                "img_split_sorted.pt".format(i + 1, j + 1)))

        # Size: N_flies x N_frames x N_channels x Height x Width
        imgs = torch.load(fname)

        # Sum over all pixels and pick the one with the maximum sum
        light_time = imgs.sum([3, 4])[0, :, 0].argmax().item()

        imgs_dicts["light_times"][i].append(light_time)
        imgs_dicts["centered_min"][i].append(
            (torch.arange(imgs_dicts["max_times"][i][j]) -
                light_time)[0].item())
        imgs_dicts["centered_max"][i].append(
            (torch.arange(imgs_dicts["max_times"][i][j]) -
                light_time)[-1].item())

with open(data_folder + "imgs_dicts.json", 'w') as f:
    json.dump(imgs_dicts, f)
