from read_json import load_from_json
from parse_cedar_objects import parse_cedar_params, make_connection
from plotting import plot_1d, plot_2d
from cedar_modules import AbsSigmoid
import nengo
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

PROBE_ALL = False
SAVE_SIMULATION = False
TAU_FACTOR = 0.5

objects, connections = load_from_json("./JSON/mental_imagery_extended.json")

model = nengo.Network()

with model:
    nengo_objects = {}
    # create the nodes
    for ob_key in objects:
        name, instance = parse_cedar_params(objects[ob_key])
        if instance.__class__.__name__ == "NeuralField":
            instance.tau *= TAU_FACTOR
        instance.make_node()
        nengo_objects[name] = instance
    # create the connections
    for connection in connections:
        make_connection(connection[0][1], connection[1][1], nengo_objects)

# the list contains all nodes that are plotted in the plotting widget of the cedar architecture
objects_to_probe = [
    "Reference Behavior.intention node",  # Reference processes
    "Reference Behavior.CoS node",
    "Reference Field & Reference Production Nodes.intention node",
    "Reference Field & Reference Production Nodes.CoS node",
    "Reference Memory Nodes & Color Field.intention node",
    "Reference Memory Nodes & Color Field.CoS node",
    "Target Behavior.intention node",  # Target processes
    "Target Behavior.CoS node",
    "Target Field & Target Production Nodes.intention node",
    "Target Field & Target Production Nodes.CoS node",
    "Reference Memory Nodes & Color Field 2.intention node",
    "Reference Memory Nodes & Color Field 2.CoS node",
    "Match Field.intention node",
    "Match Field.CoS node",
    "Relational Behavior.intention node",  # Spatial processes
    "Relational Behavior.CoS node",
    "OC Field and Spatial Production Nodes  .intention node",
    "OC Field and Spatial Production Nodes  .CoS node",
    "Condition of  Dissatisfaction .intention node",
    "Condition of  Dissatisfaction .CoS node",
    "Spatial Memory Nodes.intention node",
    "Spatial Memory nodes.CoS node",
    "Colour",  # Color attention
    "Projection",  # Attention (space)
    "Indeterminent ",  # Spatial scene representation
    "Reference",  # Reference
    "Target",  # Target
    "Object-centered ",  # Relational
    "Reference Red Memory",  # Reference color memory
    "Reference Blue Memory",
    "Reference Cyan Memory",
    "Reference Green Memory",
    "Reference Orange Memory",
    "To the left of Memory",  # Spatial relation memory
    "To the Right of Memory",
    "Above Memory",
    "Below Memory",
    "Target Red Memory",  # Target color memory
    "Target Blue Memory",
    "Target Cyan Memory",
    "Target Green Memory",
    "Target Orange Memory ",
    "Reference Red Production",  # Reference color production
    "Reference Blue Production",
    "Reference Cyan Production",
    "Reference Green Production",
    "Reference Orange Production",
    "To the left of Production",  # Spatial relation production
    "To the Right of Production",
    "Above Production",
    "Below Production",
    "Target Red Production",  # Target color production
    "Target Blue Production",
    "Target Cyan Production",
    "Target Green Production",
    "Target Orange Production",
]

with model:
    probes = {}
    for key in nengo_objects:
        if not PROBE_ALL:
            if key in objects_to_probe:
                probes[key] = nengo.Probe(
                    nengo_objects[key].node, sample_every=0.01
                )
        else:
            probes[key] = nengo.Probe(
                nengo_objects[key].node, sample_every=0.05
            )

start_time = time.time()

# build the model
# sim = nengo.Simulator(model)
with nengo.Simulator(model) as sim:
    # with sim:
    # Supply first sentence: There is a cyan object above a green object
    nengo_objects["Reference: Green"].active = True
    nengo_objects["Target: Cyan"].active = True
    nengo_objects["Spatial relation: Above"].active = True

    sim.run_steps(int(500 * TAU_FACTOR))

    # Activate imagine node
    nengo_objects["Reference: Green"].active = False
    nengo_objects["Target: Cyan"].active = False
    nengo_objects["Spatial relation: Above"].active = False
    nengo_objects["Action: Imagine"].active = True

    sim.run_steps(int(8500 * TAU_FACTOR))

    # Supply second sentence: There is a red object to the left of the green object
    nengo_objects["Reference: Green"].active = True
    nengo_objects["Target: Red"].active = True
    nengo_objects["Spatial relation: Left"].active = True

    sim.run_steps(int(500 * TAU_FACTOR))

    nengo_objects["Reference: Green"].active = False
    nengo_objects["Target: Red"].active = False
    nengo_objects["Spatial relation: Left"].active = False

    sim.run_steps(int(8500 * TAU_FACTOR))

    # Supply third sentence: There is a blue object to the right of the red object
    nengo_objects["Reference: Red"].active = True
    nengo_objects["Target: Blue"].active = True
    nengo_objects["Spatial relation: Right"].active = True

    sim.run_steps(int(500 * TAU_FACTOR))

    nengo_objects["Reference: Red"].active = False
    nengo_objects["Target: Blue"].active = False
    nengo_objects["Spatial relation: Right"].active = False

    sim.run_steps(int(8500 * TAU_FACTOR))

    # supply fourth sentence: There is an orange object to the left of the blue object
    nengo_objects["Reference: Blue"].active = True
    nengo_objects["Target: Orange"].active = True
    nengo_objects["Spatial relation: Left"].active = True

    sim.run_steps(int(500 * TAU_FACTOR))

    nengo_objects["Reference: Blue"].active = False
    nengo_objects["Target: Orange"].active = False
    nengo_objects["Spatial relation: Left"].active = False

    sim.run_steps(int(8500 * TAU_FACTOR))

    simdata = sim.data


end_time = time.time()
print(
    "\n Total time needed for simulating 4 sentences with tau_factor %.2f: %.2f minutes \n"
    % (TAU_FACTOR, (end_time - start_time) / 60)
)
# get timestamp for saving data and images
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(timestamp)

if SAVE_SIMULATION:
    os.mkdir("./simulation_data/%s" % timestamp)
    for ob_key in probes:
        file_name = ob_key.replace("/", "_")
        np.save(
            "./simulation_data/%s/%s_%s" % (timestamp, file_name, timestamp),
            simdata[probes[ob_key]],
        )

# for TAU_FACTOR 0.05 show every fifth step, for other TAU_FACTORs show a multiple
# thereof
num_samples = simdata[probes["Reference Blue Memory"]].shape[0]


if num_samples > 150:
    stepsize = 5 * round(TAU_FACTOR / 0.05)
    print("Number of samples:", num_samples, 'Stepsize:', stepsize)
    time_points = np.arange(0, num_samples, stepsize)[-36:]
    
else:
    time_points = np.arange(0, num_samples, 2)[-36:]
print("time_points: \n", time_points, len(time_points))

sigmoid = AbsSigmoid()

x = np.arange(0, num_samples) * 10

plots_1d = [
    "Target Red Memory",
    "Target Blue Memory",
    "Target Cyan Memory",
    "Target Green Memory",
    "Target Orange Memory ",
    "Target Red Production",
    "Target Blue Production",
    "Target Cyan Production",
    "Target Green Production",
    "Target Orange Production",
    "To the left of Memory",
    "To the Right of Memory",
    "Above Memory",
    "Below Memory",
    "Empty",
    "To the left of Production",
    "To the Right of Production",
    "Above Production",
    "Below Production",
    "Empty",
    "Reference Red Memory",
    "Reference Blue Memory",
    "Reference Cyan Memory",
    "Reference Green Memory",
    "Reference Orange Memory",
    "Reference Red Production",
    "Reference Blue Production",
    "Reference Cyan Production",
    "Reference Green Production",
    "Reference Orange Production",
]

# plot Memory and Production nodes
plt.figure(figsize=(15, 18))
# plt.suptitle("Memory and Production Nodes")
for i, name in enumerate(plots_1d):
    if name == "Empty":
        continue
    plt.subplot(6, 5, i + 1)
    plt.plot(x, simdata[probes[name]])
    plt.title(name)

plt.tight_layout(rect=(0, 0, 1, 0.97))

# save
if not os.path.isdir("../images/paper_example/%.2f" % TAU_FACTOR):
    os.mkdir("../images/paper_example/%.2f" % TAU_FACTOR)
plt.savefig(
    "../images/paper_example/%.2f/Memory and Production Nodes_%.2f_%s.png"
    % (TAU_FACTOR, TAU_FACTOR, timestamp)
)

# plot spatial scene
filepath = "../images/paper_example/%.2f/Spatial Scene_%.2f_%s.png" % (
    TAU_FACTOR,
    TAU_FACTOR,
    timestamp,
)
plot_2d(
    simdata[probes["Indeterminent "]],
    time_points,
    colorbar=True,
    save=filepath,
)
# plot sigmoided spatial scene
filepath = "../images/paper_example/%.2f/Sigmoided Spatial Scene_%.2f_%s.png" % (
    TAU_FACTOR,
    TAU_FACTOR,
    timestamp,
)
plot_2d(
    sigmoid(simdata[probes["Indeterminent "]]),
    time_points,
    colorbar=True,
    save=filepath,
)
# plot the colour field
filepath = "../images/paper_example/%.2f/Colour_%.2f_%s.png" % (
    TAU_FACTOR,
    TAU_FACTOR,
    timestamp,
)
plot_1d(
    simdata[probes["Colour"]],
    time_points,
    save=filepath,
)
# plot the relational field
filepath = "../images/paper_example/%.2f/Relational Field_%.2f_%s.png" % (
    TAU_FACTOR,
    TAU_FACTOR,
    timestamp,
)
object_centered_data = simdata[probes["Object-centered "]]
plot_2d(
    object_centered_data,
    time_points,
    colorbar=True,
    save=filepath,
)
# plot the target field
filepath = "../images/paper_example/%.2f/Target Field_%.2f_%s.png" % (
    TAU_FACTOR,
    TAU_FACTOR,
    timestamp,
)
target = simdata[probes["Target"]]
plot_2d(
    target,
    time_points,
    colorbar=True,
    save=filepath,
)
# plot the reference field
filepath = "../images/paper_example/%.2f/Reference Field_%.2f_%s.png" % (
    TAU_FACTOR,
    TAU_FACTOR,
    timestamp,
)
reference_data = simdata[probes["Reference"]]
plot_2d(
    reference_data,
    time_points,
    colorbar=True,
    save=filepath,
)

print("\n All plots created. \n")

