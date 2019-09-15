"""prepare_test.py
BEFORE RUNNING THIS: make sure the data is there

Prepare the test data for the neural net:
1. Crop each image to get rid of unnecessary information.
2. Run through the bilinear filter to smoothen.
3. Split each image into stripes each containing one fly.
"""

import torch
import glob
import cv2
import json

data_folder = "data/orig/"

height = 480
width = 640

imgs_dicts = {"genes": [], "nums": [], "times": [],
              "max_times": [], "times_arg_sorted": []}

# For each gene
for i in range(7):
    imgs_dicts["genes"].append(i + 1)
    imgs_dicts["nums"].append([])
    imgs_dicts["times"].append([])
    imgs_dicts["max_times"].append([])
    imgs_dicts["times_arg_sorted"].append([])

    if i == 0 or i == 1:
        size_imgs_split = 4
    else:
        size_imgs_split = 5

    # For each iteration
    for j in range(10):
        imgs_dicts["nums"][i].append(j + 1)
        imgs_dicts["times"][i].append([])
        imgs_dicts["times_arg_sorted"][i].append([])

        imgs_split = [torch.FloatTensor()] * size_imgs_split

        # For each frame
        for fname in glob.iglob(data_folder +
                                "{:02}*/*{:02}/*.jpg".format(i + 1, j + 1)):
            # Extract the frame number
            digit = int(
                "".join(list(filter(str.isdigit, fname.split("/")[-1]))))
            imgs_dicts["times"][i][j].append(digit)

            # Prepare the image (crop and filter)
            img = cv2.imread(fname)
            img = img[:, 28:540]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.bilateralFilter(img, 15, 5, 5)
            img = torch.tensor(img, dtype=torch.float)

            if ("01_Pink_impTNT" in fname) or ("02_Yellow_gfp" in fname):
                # In PINK and YELLOW experiments, the camera was slightly
                # shifted, and there were only 4 flies out of 5 possible
                # in the arena
                for k in range(4):
                    img_split = img[74 + k * height // 5:
                                    74 + (k + 1) * height // 5, :]
                    imgs_split[k] = torch.cat((imgs_split[k],
                                               img_split[None, None, :, :]), 0)
            else:
                for k in range(5):
                    img_split = img[k * height // 5:
                                    (k + 1) * height // 5, :]
                    imgs_split[k] = torch.cat((imgs_split[k],
                                               img_split[None, None, :, :]), 0)

        # Size: N_flies x N_frames x N_channels x Height x Width
        imgs_split = torch.stack(imgs_split, 0)

        # `glob.iglob` is does not extract images correctly ordered, so
        # here we order the split images
        times_arg_sorted = torch.tensor(imgs_dicts["times"][i][j]).argsort()
        imgs_dicts["times_arg_sorted"][i][j].append(times_arg_sorted.tolist())
        imgs_dicts["max_times"][i].append(max(imgs_dicts["times"][i][j]))

        imgs_split = imgs_split[:, times_arg_sorted, :, :, :]
        print(i, j, imgs_split.size())
        torch.save(
            imgs_split,
            "/".join(fname.split("/")[:-1]) + "/img_split_sorted.pt")

with open("data/orig/imgs_dicts.json", 'w') as f:
    json.dump(imgs_dicts, f)
