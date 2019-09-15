"""predict_test.py
BEFORE RUNNING THIS: run prepare_test.py, train_split.py

1. Make keypoints predictions for each fly in each iteration of each gene.
"""

import torch
import glob
from model import Net

data_folder = "data/orig/"
model_folder = "models/"

model = Net()
model.load_state_dict(torch.load(
    model_folder + "model_state_dict_cnn_2_aux_2_no-frozen.pt"))

# For each gene
for i in range(1, 8):
    # For each iteration
    for j in range(1, 11):
        fname = next(glob.iglob(
            data_folder + "{:02}*/*{:02}/img_split_sorted.pt".format(i, j)))

        # Load and normalize the input
        test_input = torch.load(fname)
        mu, std = test_input.mean(), test_input.std()
        test_input = test_input.sub(mu).div(std)

        # Set the batch size to the size of the input
        batch_size = test_input.size(1)
        model.set_batch_size(batch_size)

        outputs = torch.FloatTensor()

        # For each fly
        for test_in in test_input:
            with torch.no_grad():
                model.eval()
                _, _, output = model(test_in)
                outputs = torch.cat((outputs, output[None, :, :]), 0)

        print(i, j, outputs.size())
        torch.save(outputs, "/".join(fname.split("/")[:-1]) + "/keypoints.pt")
