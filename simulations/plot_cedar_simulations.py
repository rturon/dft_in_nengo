import pandas as pd
import glob
import os
from plotting import plot_probes_cedar

path_start = '/home/rabea/cedarRecordings/mental_imagery_extended_recording/'
dataset = "Ragni2018_smalllarge"
# SET TIME FACTOR MANUALLY, NOT ANYWHERE SAVED IN DATA!!
time_factor = 0.26

# get all paths with 3ps as part of the folder name
files = []
task_folders = glob.glob(path_start+dataset+'*')
for folder in task_folders:
    trials = glob.glob(folder+'/*')
    for trial in trials:
        trial_files = glob.glob(trial+'/*')
        files.append(trial_files)

# contains all files in subfolders that belong to dataset "dataset"
print(len(files))

# folder to save images to
save_dir = "../images/%s_cedar/" % dataset
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
save_dir = save_dir + "%.2f/" % time_factor
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
# folder structure now as it should be, for the real save_dir still need
# the task


def get_name(filepath):
    filename = filepath.rsplit('/', 1)[1]
    name = filename.rsplit('.', 2)[0].rsplit('[', 1)[0]
    name = name.replace('_', ' ')

    return name

files_to_load = ['Target Red Memory', 'Target Blue Memory', 'Target Cyan Memory',
               'Target Green Memory', 'Target Orange Memory ', 'Target Red Production',
               'Target Blue Production', 'Target Cyan Production', 'Target Green Production',
               'Target Orange Production', 'To the left of Memory', 'To the Right of Memory',
               'Above Memory', 'Below Memory', 'To the left of Production',
               'To the Right of Production', 'Above Production', 'Below Production',
               'Reference Red Memory', 'Reference Blue Memory', 'Reference Cyan Memory',
               'Reference Green Memory', 'Reference Orange Memory',
               'Reference Red Production', 'Reference Blue Production',
               'Reference Cyan Production', 'Reference Green Production',
               'Reference Orange Production',
               'Indeterminent ', 'Object-centered ', 'Reference', 'Target', 'Colour']

for trial_files in files:
    cedar_data = {}
    task_id = trial_files[0].split('/')[-3].split('_')[-9]
    print('Task_id:', task_id)
    for filepath in trial_files:
        module_name = get_name(filepath)
        if module_name == 'Match FIeld':
            continue
        if module_name in files_to_load:
            cedar_data[module_name] = pd.read_csv(filepath, skiprows=1, header=None,
                                                  error_bad_lines=False,
                                                  warn_bad_lines=False)
    
    plot_probes_cedar(cedar_data, save=save_dir+task_id)
    # plot_probes_cedar(cedar_data, )
