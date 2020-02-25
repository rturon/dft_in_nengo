from read_json import load_from_json
from parse_cedar_objects import parse_cedar_params, make_connection
import nengo


def create_model(filepath, tau_factor=0.2, sample_every=0.01):

    objects, connections = load_from_json(filepath)

    model = nengo.Network()

    with model:
        nengo_objects = {}
        # create the nodes
        for ob_key in objects:
            name, instance = parse_cedar_params(objects[ob_key])
            if instance.__class__.__name__ == "NeuralField":
                instance.tau *= tau_factor
            instance.make_node()
            nengo_objects[name] = instance
        # create the connections
        for connection in connections:
            make_connection(connection[0][1], connection[1][1], nengo_objects)

    # the list contains all nodes that are plotted in the plotting widget of the cedar architecture
    objects_to_probe = [
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
        for key in objects_to_probe:
            probes[key] = nengo.Probe(
                nengo_objects[key].node, sample_every=sample_every
            )

    return model, nengo_objects, probes


def run_task(simulator, task, nodes, tau_factor=0.2):
    for i, premise in enumerate(task):
        print("Current premise: ", premise)
        for node in premise:
            # print('Setting node:', nodes[node].name)
            nodes[node].active = True

        simulator.run_steps(int(500 * tau_factor))
        if i == 0:
            nodes["Action: Imagine"].active = True
        for node in premise:
            # print('Setting node:', nodes[node].name)
            nodes[node].active = False
        simulator.run_steps(int(7500 * tau_factor))

    simulator.close()
