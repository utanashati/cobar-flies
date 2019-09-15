"""create_dataset_template.py

Create a COCO style JSON for labeling the data.
"""

import glob
import json

keypoints = [
    "head tip",
    "body end"
]

keypoints_style = [
    "#00FF00",
    "#00FFFF"
]

categories = [{
    "id": "0",
    "name": "fly",
    "supercategory": "fly",
    "keypoints": keypoints,
    "keypoints_style": keypoints_style
}]

image_dir_regx = 'data/label/*/*.jpg'
images = []
for image_path in glob.glob(image_dir_regx):
    print(image_path)
    image_file_name = image_path[11:]
    image_id = image_file_name.replace("/", "_")
    image_url = "http://localhost:6010/" + image_file_name
    print(image_id)
    print(image_url)
    images.append({
        "id": image_id,
        "file_name": image_file_name,
        "url": image_url
    })

dataset = {
    "categories": categories,
    "images": images,
    "annotations": [],
    "licenses": []
}

dataset_file_path = "data/label/labeled_expanded.json"
with open(dataset_file_path, 'w') as f:
    json.dump(dataset, f)
