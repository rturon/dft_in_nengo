from json import JSONDecoder

def read_JSON_as_list(filename):
    ''' Reads a JSON file as a list of lists. Necessary for cedar architectures
        since they contain duplicate keys. 

        Parameters
        ----------
        filename: str
            Path to the JSON file.

        Returns
        -------
        list
            Content of JSON file as list of lists.
    '''
    def parse_pairs(pairs):
        return pairs

    decoder = JSONDecoder(object_pairs_hook=parse_pairs)
    with open(filename) as json_file:
        file_content = json_file.read()
        data = decoder.decode(file_content)
        
    return data


def fetch_objects(json_as_list):
    ''' Fetches all objects of a cedar architecture, both the ones outside
        groups and the one inside groups. For the ones inside groups the group
        name is added to the object's name to avoid duplicate object names.

        Parameters
        ----------
        json_as_list: list
            List of lists that contains a cedar architecture as returned by
            read_JSON_as_list. 

        Returns
        -------
        list
            List of all objects/elements of a cedar architecture with their 
            parameters. 
    '''
    # first get the objects outside of groups
    objects = json_as_list[1][1]
    
    key_to_pos_dict = {}
    for ind, elem in enumerate(json_as_list):
        key_to_pos_dict[elem[0]] = ind
        
    groups = json_as_list[key_to_pos_dict['groups']][1]
    
    for group in groups:
        for obj in group[1][1][1]:
            # add the group name to the object, otherwise there might be duplicate object names
            # also added in connections
            obj[1][0] = ('name', group[0] + "." + obj[1][0][1])
            objects.append(obj)
            
    object_dict = {}
    for obj in objects:
        object_dict[obj[1][0][1]] = obj
    return object_dict


def fetch_connections(json_as_list):
    ''' Fetches all connections of a cedar architecture, replacing connections
        that contain group input or group output nodes by direct connections
        from their sources to their targets. 

        Parameters
        ----------
        json_as_list: list
            List of lists that contains a cedar architecture as returned by
            read_JSON_as_list.

        Returns
        -------
        list
            List of all connections, both inside and outside of groups with 
            connections to and from group inputs and outputs replaced by direct
            connections. 
    '''
    ##########################################################################
    ####################### helper functions ################################
    def add_group_name(group_connection, group_name):
        new_group_connection = [('source', group_name + "." + group_connection[0][1]),
                                ('target', group_name + "." + group_connection[1][1])]
        return new_group_connection

    def find_connector_connections(connectors, 
                                outside_group_connections, 
                                inside_group_connections,
                                output_nodes=True):
        ''' outside_group_connections list of connections, inside_group_connections
            dictionary of list of connections per group.
        
            output_nodes True means that the connectors in the list are output
            nodes of their groups, if it is False they are input nodes.
            Elements in connectors should be of the form: 
            (group_name, connector_name)
        '''
        connector_connections = {}

        for connector in connectors:
            connector_connections[connector] = {'input connections': [], 
                            'output connections': []}

            if output_nodes:
                source_string = connector[0] + "." + connector[1]
                target_string = connector[0] + "." + connector[1] + ".input"
            else:
                source_string = connector[0] + "." + connector[1] + ".output"
                target_string = connector[0] + "." + connector[1] 

            # need the ":", otherwise removing and iterating at the same time leads to 
            # weird results
            for connection in outside_group_connections[:]:
                if connection[0][1] == source_string:
                    connector_connections[connector]['output connections'].append(connection) 
                    outside_group_connections.remove(connection)
                elif connection[1][1] == target_string:
                    connector_connections[connector]['input connections'].append(connection)
                    outside_group_connections.remove(connection)

            for connection in inside_group_connections[connector[0]][:]:
                if connection[1][1] == target_string:
                    connector_connections[connector]['input connections'].append(connection)
                    inside_group_connections[connector[0]].remove(connection)
                elif connection[0][1] == source_string:
                    connector_connections[connector]['output connections'].append(connection)
                    inside_group_connections[connector[0]].remove(connection)
                    
        return connector_connections

    def replace_middle_connection(middle_connection_dict):
        ''' takes as input a dictionary with input connections to a node and 
            output connections from the node and creates new connections substituting
            the middle node and connecting the inputs from the input nodes and the 
            outputs from the output nodes directly.
        '''
        # each input has to be connected to each output
        connections_list = []

        # go through all inputs
        for inp_connection in middle_connection_dict['input connections']:
            # source at position 0
            source = inp_connection[0]
            # go through all outputs
            for out_connection in middle_connection_dict['output connections']:
                # target at position 1
                target = out_connection[1]
                connections_list.append([source, target])

        return connections_list
    ##########################################################################
    ##########################################################################

    key_to_pos_dict = {}
    for ind, elem in enumerate(json_as_list):
        key_to_pos_dict[elem[0]] = ind
        
    connections_outside_groups = json_as_list[key_to_pos_dict["connections"]][1]
    groups = json_as_list[key_to_pos_dict["groups"]][1]

    # get the connectors (input and output nodes of groups) and all group connections
    connectors = {}
    group_connections = {}

    for group in groups:
        # get the group name
        group_name = group[0]
        key_to_pos_dict_group = {}
        for ind, elem in enumerate(group[1]):
            key_to_pos_dict_group[elem[0]] = ind
        # get the connectors of this group
        connectors[group_name] = group[1][key_to_pos_dict_group["connectors"]][1]
        # get the connections of this group and add the group name to the 
        # object's name
        add_group_name_lambda = lambda x: add_group_name(x, group_name)
        group_connections_wn = list(map(add_group_name_lambda, group[1][key_to_pos_dict_group["connections"]][1]))
        group_connections[group_name] = group_connections_wn

    # divide connectors into input and output nodes
    group_outputs = []
    group_inputs = []

    for group in groups:
        for connector in connectors[group[0]]:
            if connector[1] == 'true':
                group_inputs.append((group[0], connector[0]))
            else:
                group_outputs.append((group[0], connector[0]))

    # get all connections to and from group output nodes
    group_output_connections = find_connector_connections(group_outputs, 
                                                          connections_outside_groups,
                                                          group_connections)
    # replace source -> group_output -> target connections by source -> target connections
    replaced_connections = list(map(replace_middle_connection, 
                                    group_output_connections.values()))
    # add the replaced connections to the connection list
    _ = [connections_outside_groups.extend(cons) for cons in replaced_connections]

    # now do the same for the group input nodes
    group_input_connections = find_connector_connections(group_inputs, 
                                                          connections_outside_groups,
                                                          group_connections,
                                                          output_nodes=False)
    replaced_connections = list(map(replace_middle_connection,
                                    group_input_connections.values()))
    _ = [connections_outside_groups.extend(cons) for cons in replaced_connections]

    # finally add the connections inside the groups now that all connections
    # to group input and output nodes have been removed
    _ = [connections_outside_groups.extend(cons) for cons in group_connections.values()]

    return connections_outside_groups

def create_size_param(obj, size_value, CM=False):
    if not CM:
        sizes = ('sizes', size_value)
    else:
        # check if CM already has one size param --> DOES NOT WORK LIKE THIS!!
        # the obj here is the obj from which to get the size, not for which to
        # get the size 
        # solution: return it with 'inp size1' here and check if that's what's 
        # needed in backtrace function
        sizes = ('inp size1', size_value)

    return sizes

def get_size_param(obj, CM=False):
    # TODO: add option for checking for ComponentMultiply sizes
    # TODO: add option to add ComponentMultiply sizes instead of normal sizes
    if any(['size y' in tpl[0] for tpl in obj[1]]):
        ind = [i for i, elem in enumerate(obj[1]) if elem[0] == 'size y'][0]
        # sizes = ('sizes', [obj[1][ind-1][1],obj[1][ind][1]])
        sizes = create_size_param(obj, [obj[1][ind-1][1],obj[1][ind][1]], CM=CM)
    elif any(['output dimension sizes' in tpl[0] for tpl in obj[1]]):
        ind = [i for i, elem in enumerate(obj[1]) if elem[0] == 'output dimension sizes'][0]
        # sizes = ('sizes', obj[1][ind][1])
        sizes = create_size_param(obj, obj[1][ind][1], CM=CM)
    elif any(['sizes' in tpl[0] for tpl in obj[1]]):
        ind = [i for i, elem in enumerate(obj[1]) if elem[0] == 'sizes'][0]
        sizes = create_size_param(obj, obj[1][ind][1], CM=CM)
    elif any(['inp size' in tpl[0] for tpl in obj[1]]):
        # if the input is a ComponentMultiply then it's output size is the size
        # of its higher dimensional input
        # ind1 = [i for i, elem in enumerate(obj[1]) if elem[0] == 'inp size1'][0]
        # ind2 = [i for i, elem in enumerate(obj[1]) if elem[0] == 'inp size2'][0]
        # print('Source is a CM object:', obj[1][0][1])
        # print(obj[1])
        inp_size1 = [elem[1] for elem in obj[1] if elem[0] == 'inp size1'][0]
        inp_size2 = [elem[1] for elem in obj[1] if elem[0] == 'inp size2'][0]
        out_size = inp_size1 if len(inp_size1) >= len(inp_size2) else inp_size2
        # print('Input size 1 and 2:', inp_size1, inp_size2)
        # print('Used as output size:', out_size, '\n')
        sizes = create_size_param(obj, out_size, CM=CM)
    else:
        raise Exception('The object %s does not have a size parameter!' %obj[1][0][1])
    return sizes


def backtrace_size(obj, connections, objects_dict, obj_wo_size):
    obj_wo_sizes = ['cedar.processing.Projection', 'cedar.processing.ComponentMultiply',
                    'cedar.processing.steps.Convolution', 'cedar.processing.Flip',
                    'cedar.processing.StaticGain']
    
    # TODO: add ComponentMultiply sizes to list
    size_params = ['size x', 'size y', 'sizes', 'output dimension sizes', 
                   'inp size1', 'inp size2']

    source = None
    for connection in connections:
        # test if obj is target of connection
        if obj[1][0][1] == connection[1][1].rsplit('.',1)[0]:
            source_name = connection[0][1].rsplit('.',1)[0]
            source = objects_dict[source_name]
            # check if the source has some size parameter
            if any([size_param in object_param for size_param in size_params 
                    for object_param in source[1]]):
                if obj_wo_size[0] == "cedar.processing.ComponentMultiply":
                    sizes = get_size_param(source, CM=True)
                    if any(['inp size1' in tpl[0] for tpl in obj_wo_size[1]]):
                        sizes = ('inp size2', sizes[1])
                else:
                    sizes = get_size_param(source)
                obj_wo_size[1].append(sizes)
                return None
            
    # if no source was an object with size, go deeper, for simplicity take last source
    if source is not None:
        backtrace_size(source, connections, objects_dict, obj_wo_size)
    else:
        # The object is not really connected to the architecture since the objects without
        # a size parameter need some input to do something
        print('The object %s does not have a source!' %obj[1][0][1])

def backtrace_CM_size(cm_obj, connections, objects_dict):
    # print('Trying to backtrace size for a CM object:', cm_objget_size_param[1][0][1])
    # if the object to get the size for is a ComponentMultiply get_size_paraminstance, 
    # wee need to get the two input sizes for this object. 
    # we can assume that it has exactly two inputs, not more or less
    # but we still check if that's true
    sources = []
    # first just get the two sources
    for connection in connections:
        # test if obj is target of connection
        if cm_obj[1][0][1] == connection[1][1].rsplit('.',1)[0]:
            source_name = connection[0][1].rsplit('.',1)[0]
            source = objects_dict[source_name]
            sources.append(source)

    if len(sources) != 2:
        print('ComponentMultiply does not have 2 inputs, but %i!' %len(sources))
        return None
    
    size_params = ['size x', 'size y', 'sizes', 'output dimension sizes', 
                   'inp size1', 'inp size2']
    # now get the sizes from the sources
    if any([size_param in object_param for size_param in size_params 
            for object_param in sources[0][1]]):
        # add the size as "inp size1"
        inp_size1 = get_size_param(sources[0], CM=True)
        cm_obj[1].append(inp_size1)
    else:
        # go deeper 
        backtrace_size(sources[0], connections, objects_dict, cm_obj)
    if any([size_param in object_param for size_param in size_params 
            for object_param in sources[1][1]]):
        # add the size as "inp size2"
        inp_size2 = get_size_param(sources[1], CM=True)
        inp_size2 = ('inp size2', inp_size2[1])
        cm_obj[1].append(inp_size2)
    else:
        # go deeper
        backtrace_size(sources[1], connections, objects_dict, cm_obj)
    # we add a new size_param for the ComponentMultiplies, "inp size1" and 
    # "inp size2"


def load_from_json(filename):

    # load the file as list of lists
    json = read_JSON_as_list(filename)

    # get the objects and connections
    object_dict = fetch_objects(json)
    connections = fetch_connections(json)

    # add size parameter to objects without size parameter
    # here a slightly different size_params list without output dimension sizes
    size_params = ['size x', 'size y', 'sizes', 
                   'inp size1', 'inp size2']

    for obj in object_dict.values():
        has_size = any([size_param in object_param for size_param in size_params 
                    for object_param in obj[1]])
        if not has_size and obj[0] != 'cedar.processing.sources.Boost':
            if obj[0] != 'cedar.processing.ComponentMultiply':
                backtrace_size(obj, connections, object_dict, obj)
            else:
                backtrace_CM_size(obj, connections, object_dict)

    return object_dict, connections

