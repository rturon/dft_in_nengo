import pandas as pd
from task_creation import create_task_list
from create_model import create_model, run_task
import nengo
import matplotlib.pyplot as plt
from plotting import plot_2d, plot_1d, plot_0d
import numpy as np
from datetime import datetime
import os
import time


datapath = "../ccobra_datasets/"
dataset0 = "Ragni2018_carddir.csv"
dataset1 = "Ragni2018_smalllarge.csv"
dataset2 = "3ps.csv"
dataset3 = "4ps.csv"

tau_factor = 0.1
dataset = dataset2

image_dir = "../images/%s/" % dataset
tasks = create_task_list(datapath + dataset)

task_id_dict = {i: task for (i, task) in enumerate(tasks)}


def plot_probes(sim, probes, save=False):

    timestamp = str(datetime.now()).rsplit(".", 1)[0]
    time_points = np.linspace(
        0, sim.data[probes["Indeterminent "]].shape[0] - 1, 36, dtype=int
    )

    if save:
        filename = "_Spatial Scene_%s.png" % timestamp
        plot_2d(
            sim.data[probes["Indeterminent "]],
            time_points,
            colorbar=True,
            title=None,
            save=save + filename,
        )

        filename = "_Reference Field_%s.png" % timestamp
        plot_2d(
            sim.data[probes["Reference"]],
            time_points,
            colorbar=True,
            title=None,
            save=save + filename,
        )

        filename = "_Target Field_%s.png" % timestamp
        plot_2d(
            sim.data[probes["Target"]],
            time_points,
            colorbar=True,
            title=None,
            save=save + filename,
        )

        filename = "_Relational Field_%s.png" % timestamp
        plot_2d(
            sim.data[probes["Object-centered "]],
            time_points,
            colorbar=True,
            title=None,
            save=save + filename,
        )

        filename = "_Colour_%s.png" % timestamp
        plot_1d(
            sim.data[probes["Colour"]],
            time_points,
            title=None,
            save=save + filename,
        )

        filename = "_Memory and Production Nodes_%s.png" % timestamp
        plot_0d(sim, probes, title=None, save=save + filename)

        plt.close("all")

    else:
        plot_2d(
            sim.data[probes["Indeterminent "]],
            time_points,
            colorbar=True,
            title=None,
            save=save,
        )

        plot_2d(
            sim.data[probes["Reference"]],
            time_points,
            colorbar=True,
            title=None,
            save=save,
        )

        plot_2d(
            sim.data[probes["Target"]],
            time_points,
            colorbar=True,
            title=None,
            save=save,
        )

        plot_2d(
            sim.data[probes["Object-centered "]],
            time_points,
            colorbar=True,
            title=None,
            save=save,
        )

        plot_1d(sim.data[probes["Colour"]], time_points, title=None, save=save)

        plot_0d(sim, probes, title=None, save=False)

        plt.show()


# run tasks
start_time = time.time()

for i, task in enumerate(tasks):
    model, nodes, probes = create_model(
        "./JSON/mental_imagery_extended.json", tau_factor=tau_factor
    )
    sim = nengo.Simulator(model)
    run_task(sim, task, nodes, tau_factor=tau_factor)

    savedir = "../images/%s/%.2f/%i" % (dataset.split(".")[0], tau_factor, i)
    if not os.path.isdir("../images/%s" % dataset.split(".")[0]):
        os.mkdir("../images/%s" % dataset.split(".")[0])

    plot_probes(sim, probes, savedir)
    sim.close()

print(
    "Total time needed for tasks in %s: %.1f min"
    % (dataset.split(".")[0], (time.time() - start_time) / 60)
)

