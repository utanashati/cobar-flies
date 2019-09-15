"""plot_traces_overview.py
BEFORE RUNNING THIS: run count_backwards.py

1. Plot traces averaged throughtout the iterations and flies for each gene
   in one picture.
"""

import torch
import glob
from matplotlib import pyplot as plt

data_folder = "data/orig/"
pics_folder = "pics/"

genes = ["ImpTNT", "GFP", "ShalRNAi", "Dorsal", "TNT", "EagDN", "Hunchback"]

glob_ = True

with plt.style.context("bmh"):
    c_1 = list(plt.rcParams['axes.prop_cycle'])[1]['color']
    c_2 = list(plt.rcParams['axes.prop_cycle'])[8]['color']

    plt.fill_between(range(40, 91), -1, 1, alpha=0.2, color=c_2)

    plt.axhline(0, 0, 120, c="gray", linewidth=1.0, linestyle='--')

    # For each gene
    for i in range(7):
        fname = next(
            glob.iglob(data_folder + "{:02}*/fly_moves.pt".format(i + 1)))

        # Size: N_flies x N_iters x N_frames
        fly_moves = torch.load(fname).type(torch.float)

        mean = fly_moves.mean(1).mean(0).numpy()[2:-8]

        plt.plot(mean, label=genes[i])

    plt.ylim(-1, 1)
    locs, labels = plt.yticks(
        [-0.5, 0, 0.5], ["Backward", "Still", "Forward"], rotation=90)
    plt.xticks([0, 20, 40, 60, 80, 100], [-4, -2, 0, 2, 4, 6])
    for label in labels:
        label.set_verticalalignment('center')

    plt.ylabel("Movement")
    plt.xlabel("Seconds")
    plt.legend()

    plt.title("Movement Traces Overview")

    plt.savefig(pics_folder + "traces_overview.pdf", dpi=300)
    plt.savefig(pics_folder + "traces_overview.png", dpi=300)

    plt.close()
