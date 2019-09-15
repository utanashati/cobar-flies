"""plot_traces.py
BEFORE RUNNING THIS: run count_backwards.py

1. Plot traces for each fly averaged throughout the iterations
   (`glob_ = False`).
2. Plot traces averaged throughtout the iterations and flies for each gene
   (`glob_ = True`).
"""

import torch
import glob
from matplotlib import pyplot as plt

data_folder = "data/orig/"
pics_folder = "pics/"

titles = ["ImpTNT", "GFP", "ShalRNAi", "Dorsal", "TNT", "EagDN", "Hunchback"]

glob_ = True

with plt.style.context("bmh"):
    # For each gene
    for i in range(7):
        fname = next(
            glob.iglob(data_folder + "{:02}*/fly_moves.pt".format(i + 1)))

        # Size: N_flies x N_iters x N_frames
        fly_moves = torch.load(fname).type(torch.float)

        c_1 = list(plt.rcParams['axes.prop_cycle'])[1]['color']
        c_2 = list(plt.rcParams['axes.prop_cycle'])[8]['color']

        plt.fill_between(range(40, 91), -1, 1, alpha=0.2, color=c_2)

        plt.axhline(0, 0, 120, c="gray", linewidth=1.0, linestyle='--')

        if glob_:
            mean = fly_moves.mean(1).mean(0).numpy()[2:-8]
            std = fly_moves.std(1).mean(0).numpy()[2:-8]
            #print(len(mean))

            plt.plot(mean, c=c_1)

            plt.fill_between(
                range(len(mean)),
                mean - std, mean + std,
                alpha=0.1, color=c_1
            )
            plt.plot(mean - std, c=c_1, linewidth=0.7, alpha=0.3)
            plt.plot(mean + std, c=c_1, linewidth=0.7, alpha=0.3)

        else:
            for j in range(fly_moves.size(0)):
                mean = fly_moves.mean(1)[j].numpy()[2:-8]
                std = fly_moves.std(1)[j].numpy()[2:-8]
                #print(len(mean))

                plt.plot(mean, label=j)

                """
                plt.fill_between(
                    range(len(mean)),
                    mean - std, mean + std,
                    alpha=0.1, color=c_1)
                plt.plot(mean - std, c=c_1, linewidth=0.7, alpha=0.3)
                plt.plot(mean + std, c=c_1, linewidth=0.7, alpha=0.3)
                """

        plt.ylim(-1, 1)
        locs, labels = plt.yticks(
            [-0.5, 0, 0.5], ["Backward", "Still", "Forward"], rotation=90)
        plt.xticks([0, 20, 40, 60, 80, 100], [-4, -2, 0, 2, 4, 6])
        for label in labels:
            label.set_verticalalignment('center')

        plt.ylabel("Movement")
        plt.xlabel("Seconds")

        if glob_:
            plt.title(titles[i] + ": Global Mean ± Std")
        else:
            plt.title(titles[i])

        #plt.show()
        if glob_:
            plt.savefig(pics_folder + str(i) + "_" +
                        titles[i] + "_glob_avg_std.pdf", dpi=300)
            plt.savefig(pics_folder + str(i) + "_" +
                        titles[i] + "_glob_avg_std.png", dpi=300)
        else:
            plt.savefig(pics_folder + str(i) + "_" +
                        titles[i] + ".pdf", dpi=300)
            plt.savefig(pics_folder + str(i) + "_" +
                        titles[i] + ".png", dpi=300)
        plt.close()
