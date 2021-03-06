""" This script runs simulations of the spatial reasoning architecture on a
    CCOBRA dataset. The dataset to be used can be set here by assigning one of
    the datasets defined as 'dataset0' to 'dataset3' to the variable 'dataset'.
"""
from ccobra_task_creation import create_task_list
import sys
sys.path.append('../cedar_utils/')
from create_model import create_model, run_task
import nengo
from plotting import plot_2d, plot_1d, plot_0d, plot_probes
import os
import time
import json


datapath = "../ccobra_datasets/"
dataset0 = "Ragni2018_carddir.csv"
dataset1 = "Ragni2018_smalllarge.csv"
dataset2 = "3ps.csv"
dataset3 = "4ps.csv"

tau_factor = 0.01
dataset = dataset2

image_dir = "../images/%s/" % dataset.split(".")[0]
tasks = create_task_list(datapath + dataset)

task_id_dict = {i: task for (i, task) in enumerate(sorted(tasks))}


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(experiment_dir + "task_dict.json", "w") as file:
    json.dump(task_id_dict, file)

# run tasks
start_time = time.time()

for i, task in enumerate(sorted(tasks)):
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
time_needed = (time.time() - start_time) / 60
measure = "min" if time_needed < 121 else "h"
time_needed = time_needed if time_needed < 121 else time_needed / 60

print(
    "Total time needed for tasks in %s: %.1f %s"
    % (dataset.split(".")[0], time_needed, measure)
)
