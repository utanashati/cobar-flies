"""prepare_train.py
BEFORE RUNNING THIS:
- run crop_imgs_train.py (ONLY if the imgsaren't cropped already)
- run extract_keypoints.py

Prepare the train data for the neural net:
1. Augment the dataset.
   - Flip the images
        - Vertically
        - Horizontally
2. Split each image into stripes each containing one fly.
3. Split the corresponding keypoints.
"""

import cv2
import json
import torch
import glob
import numpy as np
from matplotlib import pyplot as plt

data_folder = "data/label/"

# DATA AUGMENTATION
with open(data_folder + "flies_expanded_keypoints.json", 'r') as f:
    keypoints = json.load(f)

keypoints_aug = keypoints.copy()

for key in keypoints.keys():
    img = cv2.imread(data_folder + key, 0)

    # Flip the images
    hflip = cv2.flip(img.copy(), 1)
    vflip = cv2.flip(img.copy(), 0)
    hvflip = cv2.flip(hflip.copy(), 0)
    
    cv2.imwrite(data_folder + key[:-4] + "_v.jpg", vflip)
    cv2.imwrite(data_folder + key[:-4] + "_h.jpg", hflip)
    cv2.imwrite(data_folder + key[:-4] + "_hv.jpg", hvflip)

    # Flip the corresponding keypoints
    keypoints_copy = keypoints[key].copy()
    target = torch.tensor(keypoints_copy, dtype=torch.float)
    target_v = torch.tensor(keypoints_copy, dtype=torch.float)
    target_h = torch.tensor(keypoints_copy, dtype=torch.float)
    target_hv = torch.tensor(keypoints_copy, dtype=torch.float)
    
    for i in range(2):
        for j in range(5):
            target_v[i, j, 1] = 1 - keypoints[key][i][j][1]
            target_h[i, j, 0] = 1 - keypoints[key][i][j][0]
            target_hv[i, j, 1] = 1 - keypoints[key][i][j][1]
            target_hv[i, j, 0] = 1 - keypoints[key][i][j][0]
    idx = [i for i in range(target_v.size(1)-1, -1, -1)]
    idx = torch.LongTensor(idx)
    target_v = target_v.index_select(1, idx)
    target_hv = target_hv.index_select(1, idx)

    if ("01_Pink_impTNT" in key) or ("02_Yellow_gfp" in key):
        target = target[:, 1:, :]
        target_h = target_h[:, 1:, :]

        # Uncomment to see the flipped imgs and corresponding keypoints
        """
        fig, axs = plt.subplots(2, 2)
        axs[0][0].imshow(img)
        axs[0][1].imshow(hflip)
        axs[1][0].imshow(vflip)
        axs[1][1].imshow(hvflip)

        axs[0][0].scatter(target[:, 0, 0] * 512, target[:, 0, 1] * 480, label="0")
        axs[0][0].scatter(target[:, 1, 0] * 512, target[:, 1, 1] * 480, label="1")
        axs[0][0].scatter(target[:, 2, 0] * 512, target[:, 2, 1] * 480, label="2")
        axs[0][0].scatter(target[:, 3, 0] * 512, target[:, 3, 1] * 480, label="3")
        #axs[0][0].scatter(target[:, 4, 0] * 512, target[:, 4, 1] * 480, label="4")

        axs[0][1].scatter(target_h[:, 0, 0] * 512, target_h[:, 0, 1] * 480, label="0")
        axs[0][1].scatter(target_h[:, 1, 0] * 512, target_h[:, 1, 1] * 480, label="1")
        axs[0][1].scatter(target_h[:, 2, 0] * 512, target_h[:, 2, 1] * 480, label="2")
        axs[0][1].scatter(target_h[:, 3, 0] * 512, target_h[:, 3, 1] * 480, label="3")
        #axs[0][1].scatter(target_h[:, 4, 0] * 512, target_h[:, 4, 1] * 480, label="4")

        axs[1][0].scatter(target_v[:, 0, 0] * 512, target_v[:, 0, 1] * 480, label="0")
        axs[1][0].scatter(target_v[:, 1, 0] * 512, target_v[:, 1, 1] * 480, label="1")
        axs[1][0].scatter(target_v[:, 2, 0] * 512, target_v[:, 2, 1] * 480, label="2")
        axs[1][0].scatter(target_v[:, 3, 0] * 512, target_v[:, 3, 1] * 480, label="3")
        axs[1][0].scatter(target_v[:, 4, 0] * 512, target_v[:, 4, 1] * 480, label="4")

        axs[1][1].scatter(target_hv[:, 0, 0] * 512, target_hv[:, 0, 1] * 480, label="0")
        axs[1][1].scatter(target_hv[:, 1, 0] * 512, target_hv[:, 1, 1] * 480, label="1")
        axs[1][1].scatter(target_hv[:, 2, 0] * 512, target_hv[:, 2, 1] * 480, label="2")
        axs[1][1].scatter(target_hv[:, 3, 0] * 512, target_hv[:, 3, 1] * 480, label="3")
        axs[1][1].scatter(target_hv[:, 4, 0] * 512, target_hv[:, 4, 1] * 480, label="4")

        #plt.legend()
        axs[0][0].set_title("img")
        axs[0][1].set_title("hflip")
        axs[1][0].set_title("vflip")
        axs[1][1].set_title("hvflip")
        plt.show()
        """

    keypoints_aug[key] = target.tolist()
    keypoints_aug[key[:-4] + "_v.jpg"] = target_v.tolist()
    keypoints_aug[key[:-4] + "_h.jpg"] = target_h.tolist()
    keypoints_aug[key[:-4] + "_hv.jpg"] = target_hv.tolist()

keypoints_path = "data/label/flies_expanded_keypoints_augented.json"
with open(keypoints_path, 'w') as f:
    json.dump(keypoints_aug, f)
print("Augmentation complete")

# SPLITTING INTO STRIPES
height = 480
width = 640

imgs_split = torch.FloatTensor()
targets_split = torch.FloatTensor()

for key in keypoints_aug.keys():
    img = torch.tensor(cv2.imread(data_folder + key, 0), dtype=torch.float)
    target = torch.tensor(keypoints_aug[key], dtype=torch.float)

    if ("01_Pink_impTNT" in key) or ("02_Yellow_gfp" in key):
        # In PINK and YELLOW experiments, the camera was slightly shifted,
        # and there were only 4 flies out of 5 possible in the arena
        for i in range(4):
            target_split = torch.FloatTensor(2, 2)

            if ("_h" in key) or ("_hv" in key):
                #print("h")
                target_split[:, 0] = (target[:, i, 0] * width - 100) / 512
            else:
                #print("not h")
                target_split[:, 0] = (target[:, i, 0] * width - 28) / 512

            if ("_v" in key) or ("_hv" in key):
                #print("v")
                img_split = img[20 + i * height // 5:20 + (i + 1) * height // 5, :]
                target_split[:, 1] = (target[:, i, 1] * height - 20 - height // 5 * i) / (height // 5)
            else:
                #print("not v")
                img_split = img[74 + i * height // 5:74 + (i + 1) * height // 5, :]
                target_split[:, 1] = (target[:, i, 1] * height - 74 - height // 5 * i) / (height // 5)

            if (target_split < 0).sum() + (target_split > 1).sum() != 0:
                pass
            else:
                imgs_split = torch.cat((imgs_split, img_split[None, None, :, :]), 0)
                targets_split = torch.cat((targets_split, target_split[None, :, :]), 0)

            # Uncomment to see each stripe and the original img
            """
            plt.imshow(img_split)
            plt.scatter(target_split[:, 0] * 512, target_split[:, 1] * 96)
            plt.show()
            plt.imshow(img)
            plt.scatter(target[:, 0, 0] * 512, target[:, 0, 1] * 480, label="0")
            plt.scatter(target[:, 1, 0] * 512, target[:, 1, 1] * 480, label="1")
            plt.scatter(target[:, 2, 0] * 512, target[:, 2, 1] * 480, label="2")
            plt.scatter(target[:, 3, 0] * 512, target[:, 3, 1] * 480, label="3")
            #plt.scatter(target[:, 4, 0] * 512, target[:, 4, 1] * 480, label="4")
            plt.legend()
            plt.show()
            """

    else:
        for i in range(5):
            img_split = img[i * height // 5:(i + 1) * height // 5, :]
            target_split = torch.FloatTensor(2, 2)

            if ("_h" in key) or ("_hv" in key):
                #print("h")
                target_split[:, 0] = (target[:, i, 0] * width - 100) / 512
            else:
                target_split[:, 0] = (target[:, i, 0] * width - 28) / 512

            target_split[:, 1] = (target[:, i, 1] * height - height // 5 * i) / (height // 5)

            if (target_split < 0).sum() + (target_split > 1).sum() != 0:
                print("peow")
            else:
                imgs_split = torch.cat((imgs_split, img_split[None, None, :, :]), 0)
                targets_split = torch.cat((targets_split, target_split[None, :, :]), 0)

            # Uncomment to see each stripe and the original img
            """
            plt.imshow(img_split)
            plt.scatter(target_split[:, 0] * 512, target_split[:, 1] * 96)
            plt.show()
            plt.imshow(img)
            plt.scatter(target[:, 0, 0] * 512, target[:, 0, 1] * 480, label="0")
            plt.scatter(target[:, 1, 0] * 512, target[:, 1, 1] * 480, label="1")
            plt.scatter(target[:, 2, 0] * 512, target[:, 2, 1] * 480, label="2")
            plt.scatter(target[:, 3, 0] * 512, target[:, 3, 1] * 480, label="3")
            plt.scatter(target[:, 4, 0] * 512, target[:, 4, 1] * 480, label="4")
            plt.legend()
            plt.show()
            """

print(imgs_split.size())
print(targets_split.size())
torch.save(targets_split, data_folder + "targets_expanded_split.pt")
torch.save(imgs_split, data_folder + "imgs_expanded_split.pt")
print("Split complete")
