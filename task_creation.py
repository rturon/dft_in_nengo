import pandas as pd
import os
import json

_pos_to_color = [['Target: Red', 'Reference: Red'],
                 ['Target: Blue', 'Reference: Blue'],
                 ['Target: Cyan', 'Reference: Cyan'],
                 ['Target: Green', 'Reference: Green'],
                 ['Target: Orange', 'Reference: Orange']]
_colors = ['Red', 'Blue', 'Cyan', 'Green', 'Orange']


def translate_task(task_orig):
    # create a dictionary with translations for all known spatial relations and
    # the target and reference objects A, B, C, D
    trans_dict = {'Left': 'Spatial relation: Left',
                  'Right': 'Spatial relation: Right', 'south': 'Spatial relation: Below',
                  'north': 'Spatial relation: Above', 'west': 'Spatial relation: Left',
                  'south-west': 'Spatial relation: South-West',
                  'north-west': 'Spatial relation: North-West',
                  'east': 'Spatial relation: Right',
                  'south-east': 'Spatial relation: South-East',
                  'north-east': 'Spatial relation: North-East',
                  'West': 'Spatial relation: Left',
                  'A': {'target': 'Target: Red', 'reference': 'Reference: Red'},
                  'B': {'target': 'Target: Blue', 'reference': 'Reference: Blue'},
                  'C': {'target': 'Target: Cyan', 'reference': 'Reference: Cyan'},
                  'D': {'target': 'Target: Green', 'reference': 'Reference: Green'}}

    # add other objects to the translation dictionary if needed

    objects = list(set([task_orig[j][i]
                        for j in range(len(task_orig)) for i in [1, 2]]))
    if not('A' in objects or 'B' in objects or 'C' in objects or 'D' in objects):
        trans_back = {}
        for i, ob in enumerate(objects):
            trans_dict[ob] = {'target': _pos_to_color[i]
                              [0], 'reference': _pos_to_color[i][1]}
            trans_back[_colors[i]] = ob

    task = [(trans_dict[premise[0]],
             trans_dict[premise[1]]['target'],
             trans_dict[premise[2]]['reference']) for premise in task_orig]
    task = tuple(task)
    return task


def task_to_list(task_as_str):
    premises = task_as_str.split('/')
    if premises[-1] == '':
        premises = premises[:-1]
    task = [premise.split(';') for premise in premises]
    task = tuple(task)
    return task


def create_task_list(filepath):
    data = pd.read_csv(filepath)

    tasks = data[data['id'] == 1]['task']
    task_num_or = len(tasks)
    tasks = tasks.drop_duplicates()
    tasks = list(map(task_to_list, tasks))
    tasks_translated = list(map(translate_task, tasks))
    tasks_translated = list(set(tasks_translated))
    print('Number of tasks originally: %i \n After removing duplicates: %i'
          % (task_num_or, len(tasks_translated)))

    return tasks_translated


def create_experiment(task, save_to):

    target1 = '"%s"' % task[0][1]
    target2 = '"%s"' % task[1][1]
    reference1 = '"%s"' % task[0][2]
    reference2 = '"%s"' % task[1][2]
    relation1 = '"%s"' % task[0][0]
    relation2 = '"%s"' % task[1][0]
    # check which template is needed
    if len(task) == 2:
        with open('./JSON/2PremisesTemplate.json', 'r') as file:
            fmt_str = file.read()

        experiment = fmt_str.format(target1=target1, relation1=relation1,
                                    reference1=reference1, target2=target2,
                                    relation2=relation2, reference2=reference2)

        with open(save_to, 'w') as file:
            # json.dump(experiment, file)
            file.write(experiment)

    elif len(task) == 3:
        with open('./JSON/3PremisesTemplate.json', 'r') as file:
            fmt_str = file.read()

        target3 = '"%s"' % task[2][1]
        reference3 = '"%s"' % task[2][2]
        relation3 = '"%s"' % task[2][0]

        experiment = fmt_str.format(target1=target1, relation1=relation1,
                                    reference1=reference1, target2=target2,
                                    relation2=relation2, reference2=reference2,
                                    target3=target3, relation3=relation3,
                                    reference3=reference3)

        with open(save_to, 'w') as file:
            # json.dump(experiment, file)
            file.write(experiment)

    elif len(task) == 4:
        with open('./JSON/4PremisesTemplate.json', 'r') as file:
            fmt_str = file.read()

        target3 = '"%s"' % task[2][1]
        reference3 = '"%s"' % task[2][2]
        relation3 = '"%s"' % task[2][0]
        target4 = '"%s"' % task[3][1]
        reference4 = '"%s"' % task[3][2]
        relation4 = '"%s"' % task[3][0]

        experiment = fmt_str.format(target1=target1, relation1=relation1,
                                    reference1=reference1, target2=target2,
                                    relation2=relation2, reference2=reference2,
                                    target3=target3, relation3=relation3,
                                    reference3=reference3, target4=target4,
                                    relation4=relation4, reference4=reference4)

        with open(save_to, 'w') as file:
            # json.dump(experiment, file)
            file.write(experiment)

    else:
        print('No template for %i premises!' % (len(task)))
