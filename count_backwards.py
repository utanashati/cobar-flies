"""count_backwards.py
BEFORE RUNNING THIS: run predict_test.py

1. Center the frames around the moment of tirning the light on.
2. Calculate the amount of moves forward, backward and no move (still)
   for each gene.
"""

import torch
import glob
import json
from matplotlib import pyplot as plt

data_folder = "data/orig/"
diff = 10
epsilon = 1e-3

with open("data/orig/imgs_dicts.json", 'r') as f:
    imgs_dicts = json.load(f)

centered_max = torch.tensor(imgs_dicts["centered_max"])
centered_max = centered_max.min()
centered_min = torch.tensor(imgs_dicts["centered_min"])
centered_min = centered_min.max()
#print(centered_max, centered_min)

fly_moves = []

# For each gene
for i in range(7):
    fly_moves.append([])

    # For each iteration
    for j in range(10):
        fly_moves[i].append([])

        # For centering
        light_time = imgs_dicts["light_times"][i][j]
        idx_min = light_time + centered_min
        idx_max = light_time + centered_max

        fname_keypoints = next(glob.iglob(
            data_folder + "{:02}*/*{:02}/keypoints.pt".format(i + 1, j + 1)))
        fname_imgs = next(glob.iglob(
            data_folder +
            "{:02}*/*{:02}/img_split_sorted.pt".format(i + 1, j + 1)))

        # Load centered data
        keypoints = torch.load(fname_keypoints)[:, idx_min:idx_max, :]
        imgs = torch.load(fname_imgs)[:, idx_min:idx_max, :, :, :]

        for fly_no, (fly_kps, fly_imgs) in enumerate(zip(keypoints, imgs)):
            fly_moves[i][j].append([])

            fly_forw = 0
            fly_backw = 0
            fly_still = 0
            for k, time in enumerate(range(diff, keypoints.size(1))):
                # Uncomment to see the predicted head and bottom points
                # for each fly
                """
                plt.imshow(fly_imgs[k, 0, :, :])
                plt.scatter(fly_kps[k, 0] * fly_imgs.size(3),
                            fly_kps[k, 1] * fly_imgs.size(2), label="head")
                plt.scatter(fly_kps[k, 2] * fly_imgs.size(3),
                            fly_kps[k, 3] * fly_imgs.size(2), label="bottom")
                plt.show()
                """

                centroid_curr = (fly_kps[time, 0] + fly_kps[time, 2]) / 2
                centroid_prev = (fly_kps[time - diff, 0] + fly_kps[time - diff, 2]) / 2
                centr_displacement = centroid_curr - centroid_prev

                if (centr_displacement * (fly_kps[time, 0] - fly_kps[time, 2])) > epsilon:
                    # Fly moved forward
                    fly_moves[i][j][fly_no].append(1)
                    fly_forw += 1
                elif (centr_displacement * (fly_kps[time, 0] - fly_kps[time, 2])) < -epsilon:
                    # Fly moved backward
                    fly_moves[i][j][fly_no].append(-1)
                    fly_backw += 1
                else:
                    fly_moves[i][j][fly_no].append(0)
                    fly_still += 1

            # Uncomment to see the individual traces for each fly
            """
            print("Forward: " + str(fly_forw) + "\nBackward: " +
                   str(fly_backw) + "\nStill: " + str(fly_still))
            plt.title("Fly No " + str(fly_no + 1))
            plt.scatter(
                range(len(fly_moves[i][j][fly_no])), fly_moves[i][j][fly_no])
            plt.show()
            """

    fly_moves_i = torch.tensor(fly_moves[i])

    # Size: N_flies x N_iters x N_frames
    fly_moves_i = fly_moves_i.permute(1, 0, 2)

    torch.save(fly_moves_i,
               "/".join(fname_imgs.split("/")[:-2]) + "/fly_moves.pt")
