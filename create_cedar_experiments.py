from task_creation import create_experiment, create_task_list
from create_model import create_model, run_task
from plotting import plot_2d, plot_1d, plot_0d, plot_probes
from datetime import datetime
import os
import time
import json


datapath = "../ccobra_datasets/"
dataset0 = "Ragni2018_carddir.csv"
dataset1 = "Ragni2018_smalllarge.csv"
dataset2 = "3ps.csv"
dataset3 = "4ps.csv"

tau_factor = 0.1
dataset = dataset0

experiment_dir = "../experiments/%s/" % dataset.split('.')[0]
if not os.path.isdir(experiment_dir):
    os.mkdir(experiment_dir)

tasks = create_task_list(datapath + dataset)

task_id_dict = {i: task for (i, task) in enumerate(tasks)}

with open(experiment_dir+"task_dict.json", 'w') as file:
    json.dump(task_id_dict, file)

for i, task in enumerate(tasks):
    create_experiment(task, experiment_dir+"%i_task.json" % i)
