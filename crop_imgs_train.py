"""crop_imgs_train.py
BEFORE RUNNING THIS: run extract_keypoints.py

Crop and filter the labeled images and save them to a separate folder.
"""

import glob
import json
import cv2

data_folder_label = "data/label/"
data_folder_orig = "data/orig/"

with open(data_folder_label + "ids_to_names_expanded.json", 'r') as f:
    ids_to_names = json.load(f)

for name in glob.iglob(data_folder_label + "*.jpg"):
    folder, img_name = name.split("/")
    if folder == "02_Yellow_gfp":
        if ("_128." in img_name) or ("_15." in img_name) or ("_44." in img_name):
            img_path = data_folder_orig + folder + "/" + folder.lower() + "_" + folder[:2] + "/" + img_name
        else:
            img_path = data_folder_orig + folder + "/" + folder.lower() + "_" + "01" + "/" + img_name
    else:
        img_path = data_folder_orig + folder + "/" + folder.lower() + "_" + folder[:2] + "/" + img_name
    img = cv2.imread(img_path)
    img = img[:, 28:540]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(img, 15, 5, 5)
    cv2.imwrite(data_folder_label + name, filt)
