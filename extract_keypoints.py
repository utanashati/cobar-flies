"""extract_keypoints.py
BEFORE RUNNING THIS: get the labeled dataset

The initial labeled dataset in COCO style, so we need to extract only
the necessary information.

1. Create a dictionary to convert ids into filenames.
2. Create the dataset in format {<file_name>: <keypoints>}
"""

import json

dataset_file_path = "data/label/flies_expanded_annotated.json"
with open(dataset_file_path, 'r') as f:
    dataset = json.load(f)

###############################################################################
# Create a dictionary to convert ids into filenames

dataset_ids_to_names = {}
for img in dataset["images"]:
    dataset_ids_to_names[img["id"]] = img["file_name"]

dataset_file_path = "data/label/ids_to_names_expanded.json"
with open(dataset_file_path, 'w') as f:
    json.dump(dataset_ids_to_names, f)

###############################################################################
# Create the dataset in format {<file_name>: <keypoints>}

dataset_names_keypoints = {dataset_ids_to_names[ann["image_id"]]: [[], []] for ann in dataset["annotations"]}
for ann in dataset["annotations"]:
    dataset_names_keypoints[dataset_ids_to_names[ann["image_id"]]][0].append(ann["keypoints"][:2])
    dataset_names_keypoints[dataset_ids_to_names[ann["image_id"]]][1].append(ann["keypoints"][3:5])

for key in dataset_names_keypoints.keys():
    if len(dataset_names_keypoints[key][0]) == 4:
        dataset_names_keypoints[key][0].insert(0, [0.0, 0.0])
        dataset_names_keypoints[key][1].insert(0, [0.0, 0.0])

dataset_file_path = "data/label/flies_expanded_keypoints.json"
with open(dataset_file_path, 'w') as f:
    json.dump(dataset_names_keypoints, f)
