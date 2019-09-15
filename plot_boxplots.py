"""plot_boxplots.py
BEFORE RUNNING THIS: run count_backwards.py

1. Plot boxplots with the Backward / (Forward + Backward) ratio distributions
   for each gene.
"""

import torch
import glob
from matplotlib import pyplot as plt

data_folder = "data/orig/"
pics_folder = "pics/"

genes = ["ImpTNT", "GFP", "ShalRNAi", "Dorsal", "TNT", "EagDN", "Hunchback"]
colors = ["pink", "lemonchiffon", "lightcoral", "ivory",
          "navajowhite", "lightgreen", "skyblue"]

with plt.style.context("bmh"):
    ratios = []

    # For each gene
    for i in range(7):
        fname = next(
            glob.iglob(data_folder + "{:02}*/fly_moves.pt".format(i + 1)))

        # Size: N_flies x N_iters x N_frames
        fly_moves = torch.load(fname).type(torch.float)

        # Only pick the frames with the light on
        fly_moves = fly_moves[:, :, 40:91]

        # Count the number of forward and backward moves
        fly_moves_fwd = (fly_moves == 1).sum(2).type(torch.float)
        fly_moves_bwd = (fly_moves == -1).sum(2).type(torch.float)

        ratio = (fly_moves_bwd / (fly_moves_fwd + fly_moves_bwd)).view(-1)
        ratios.append(ratio.numpy())

    box = plt.boxplot(ratios, patch_artist=True)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(range(1, 8), genes, rotation=30)
    plt.subplots_adjust(bottom=0.12)
    plt.ylabel("Backward / (Forward + Backward)")
    plt.title("Backward vs Forward Walking")
    #plt.show()
    plt.savefig(pics_folder + "boxplots.pdf", dpi=300)
    plt.savefig(pics_folder + "boxplots.png", dpi=300)
    plt.close()
