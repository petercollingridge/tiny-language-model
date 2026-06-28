import html
import os

from vis.draw_svg import SVG
from utils import parse_model_data

MARGIN_X = 100
MARGIN_Y = 10
NODE_RADIUS = 20
NODE_DX = 220
NODE_DY = 55

BLUE = [0, 60, 90]
GREY = [200, 200, 200]
YELLOW = [255, 180, 0]

RGB_COLORS = [
    [255, 0, 0],   # BR = white (not actually drawn)
    [0, 60, 128],      # are = blue
    [0, 255, 255],     # herbivores = cyan
    [255, 0, 175],     # sheep = pink
    [255, 255, 0]      # slow = yellow
]

def get_filepath(folder, filename, suffix=None):
    if suffix is not None:
        filename = f"{filename}_{suffix}"
    return os.path.join(folder, filename + ".txt")


def get_network(layout, tokens, weights):
    """
    Create a network layout which is a dictionary with nodes, edges, and layout keys.
    Nodes and edges are flat lists of dictionaries, and layout is a list of integers
    representing the number of nodes in each layer.
    """

    nodes = []
    node_layout = []
    edges = []
    x = MARGIN_X + NODE_RADIUS
    max_layer_size = max(layout)

    for layer_n, layer in enumerate(layout):
        nodes_in_layer = []
        start_y = MARGIN_Y + NODE_RADIUS + (max_layer_size - layer) * NODE_DY / 2

        # Create nodes
        for i in range(layer):
            y = start_y + i * NODE_DY
            node = { 'id': len(nodes), 'x': x, 'y': y, 'layer': layer_n, 'edges': [] }
            if layer_n in (0, len(layout) - 1):
                node['label'] = tokens[i]
            nodes_in_layer.append(node)
            nodes.append(node)

        node_layout.append(nodes_in_layer)
        x += NODE_DX

    # input layer to first hidden layer
    nodes_in_input_layer = layout[0]
    for i, row in enumerate(weights[0]):
        start_node = i
        for j, weight in enumerate(row):
            edge = { 'node1': start_node, 'node2': nodes_in_input_layer + j, 'weight': weight }
            edges.append(edge)

    start_of_output_layer = sum(layout[:-1])
    if len(layout) > 2:
        # hidden layer to output layer
        for i, row in enumerate(weights[1]):
            end_node = start_of_output_layer + i
            for j, weight in enumerate(row):
                start_node = nodes_in_input_layer + j
                edge = { 'node1': start_node, 'node2': end_node, 'weight': weight }
                edges.append(edge)

    return { 'nodes': nodes, 'edges': edges, 'layout': layout }


def get_network_2(data):
    """
    Create a network layout which is a dictionary with nodes, edges, and layout keys.
    Nodes and edges are flat lists of dictionaries, and layout is a list of list of node indices.
    Assume that we have a single hidden layer that includes a bias node.
    """

    weights = data['weights']
    tokens = data['tokens']

    nodes = []
    edges = []
    layout = []

    def add_layer(n, add_tokens=False):
        layer = []
        for i in range(n):
            _id = len(nodes)
            node = { 'id': _id, 'edges_in': [], 'edges_out': [] }
            if add_tokens:
                node['label'] = tokens[i]
            nodes.append(node)
            layer.append(_id)
        layout.append(layer)

    # Input layer
    add_layer(len(weights[0]), add_tokens=True)

    # Hidden layer
    add_layer(len(weights[0][0]) + 1)  # +1 for bias node

    # Output layer
    add_layer(len(weights[1]), add_tokens=True)

    # Create edges from input layer to hidden layer
    for i, row in enumerate(weights[0]):
        for j, weight in enumerate(row):
            node1 = i
            node2 = layout[1][j]
            edge = { 'node1': node1, 'node2': node2, 'weight': weight }
            edges.append(edge)
            nodes[node1]['edges_out'].append(edge)
            nodes[node2]['edges_in'].append(edge)

    # Create edges from hidden layer to output layer
    for i, row in enumerate(weights[1]):
        for j, weight in enumerate(row):
            node1 = layout[1][j]
            node2 = layout[2][i]
            edge = { 'node1': node1, 'node2': node2, 'weight': weight }
            edges.append(edge)
            nodes[node1]['edges_out'].append(edge)
            nodes[node2]['edges_in'].append(edge)

    # Create bias edges from hidden layer to output layer
    for i, row in enumerate(weights[2]):
        weight = row[0]
        node1 = layout[1][-1]  # Bias node is the last node in hidden layer
        node2 = layout[2][i]
        edge = { 'node1': node1, 'node2': node2, 'weight': weight }
        edges.append(edge)
        nodes[node1]['edges_out'].append(edge)  # Bias node is the last node in hidden layer
        nodes[node2]['edges_in'].append(edge)

    return { 'nodes': nodes, 'edges': edges, 'layout': layout }


def get_activation_pattern(network, layout, softmax=True):
    """
    Return a list where each item represents the activation pattern of the network for a given input token.
    The activation pattern is a dictionary of node and edge activation values.
    """

    nodes = network['nodes']
    edges = network['edges']
    activation_patterns = []


    # For each initial input token, calculate which nodes and edges are activated based on the
    # weights.
    for token in range(layout[0]):
        node_activations = [0] * len(nodes)
        edge_activations = [0] * len(edges)

        # Activate the input node corresponding to the token.
        node_activations[token] = 1

        # For each subsequent layer, calculate the activations of the nodes and edges based on
        # the previous layer's activations and the weights.
        for layer_n in range(1, len(layout)):
            start_node = sum(layout[:layer_n])
            end_node = sum(layout[:layer_n + 1])

            # For each node in the current layer, calculate its activation based on the
            # activations of the previous layer's nodes and the weights of the edges connecting them.
            for node_index in range(start_node, end_node):
                node_activation = 0
                for edge_index, edge in enumerate(edges):
                    if edge['node2'] == node_index:
                        node_activation += node_activations[edge['node1']] * edge['weight']
                node_activations[node_index] = node_activation


        # For each edge, if the starting node is activated, activate the edge and the ending node
        # based on the weight.
        for i, edge in enumerate(edges):
            if edge['node1'] == token:
                edge_activations[i] = 1
                node_activations[edge['node2']] += edge['weight']

        # Softmax the activations of the output layer nodes.
        output_layer_start = sum(layout[:-1])
        output_activations = node_activations[output_layer_start:]
        if softmax:
            max_activation = max(output_activations)
            exp_activations = [pow(2.71828, a - max_activation) for a in output_activations]
            sum_exp = sum(exp_activations)
            if sum_exp > 0:
                output_activations = [a / sum_exp for a in exp_activations]

        for i, output_activation in enumerate(output_activations):
            node_activations[output_layer_start + i] = round(output_activation, 2)

        activation_patterns.append({'nodes': node_activations, 'edges': edge_activations})

    return activation_patterns


def _add_styles(svg):
    svg.add_style('.node circle', {'fill': 'none', 'stroke': '#111', 'stroke-width': 1})
    svg.add_style('.node .active circle', {'fill': 'rgb(0, 63, 92)', 'stroke': 'rgb(0, 63, 92)', 'stroke-width': 1})
    svg.add_style('.node text', {'dominant-baseline': 'middle'})
    svg.add_style('.node .active text', {'fill': 'rgb(255, 0, 175)'})
    svg.add_style('.node .deactive text', {'fill': 'rgb(200, 200, 200)'})
    svg.add_style('.input-node', {'text-anchor': 'end'})
    svg.add_style('.output-node', {'text-anchor': 'start'})
    svg.add_style('.node text.activation-value', {
        'text-anchor': 'middle',
        'font-size': '10px',
        'fill': '#fff',
        'opacity': 0
    })
    svg.add_style('.output-node.deactive text', {'opacity': 0})
    svg.add_style('.node .active text.activation-value', {'opacity': 1})
    svg.add_style('.edge line', {'stroke-width': 2, 'stroke': 'currentColor', 'marker-end': 'url(#arrow)'})
    svg.add_style('.hit-box', {'opacity': 0.1})


def _add_arrow_marker(svg):
    defs = svg.add('defs')
    marker = defs.add('marker', {
        'id': 'arrow',
        'viewBox': '0 0 15 15',
        'refX': '5',
        'refY': '5',
        'markerWidth': '6',
        'markerHeight': '6',
        'orient': 'auto-start-reverse'
    })
    marker.add('path', {'d': 'M 0 0 L 10 5 L 0 10 z', 'fill': 'currentColor'})


def _add_script(svg, activations, filename):
    filepath = os.path.join('vis', 'js_scripts', filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        script = f.read()

    id_code = f'\nconst svgId = "{svg.attributes["id"]}";\n'
    edge_code = f'\nconst activations = {activations};\n\n'
    svg.add('script', {}, id_code + edge_code + html.escape(script))


def lerp_colour(weight, max_weight, colour1, colour2):
    ratio = (weight / max_weight) ** 1 if max_weight != 0 else 0
    return [
        int(colour1[i] + ratio * (colour2[i] - colour1[i]))
        for i in range(3)
    ]


def offset_line(x1, y1, x2, y2, offset, offset2 = None):
    if offset2 is None:
        offset2 = offset
    dx = x2 - x1
    dy = y2 - y1
    d = (dx ** 2 + dy ** 2) ** 0.5
    if d > 0:
        offset_x = dx / d * offset
        offset_y = dy / d * offset
        offset_x2 = dx / d * offset2
        offset_y2 = dy / d * offset2
        return x1 + offset_x, y1 + offset_y, x2 - offset_x2, y2 - offset_y2
    else:
        return x1, y1, x2, y2


def draw_network_svg(svg_id, token_list, layout, network):
    """ Draw a fully connected network of nodes representing the tokens in token_list. """

    n_tokens = len(token_list)
    n_layers = len(layout)

    svg_width = 2 * (MARGIN_X + NODE_RADIUS) + (n_layers - 1) * NODE_DX
    svg_height = 2 * (MARGIN_Y + NODE_RADIUS) + (n_tokens - 1) * NODE_DY

    svg = SVG({'id': svg_id, 'viewBox': f"0 0 {svg_width} {svg_height}"})
    _add_styles(svg)

    svg.rect(0, 0, svg_width, svg_height, classname='hit-box background')

    nodes_group = svg.add('g', {'class': 'node'})
    edges_group = svg.add('g', {'class': 'edge'})
    nodes = network['nodes']

    for node in nodes:
        x = node['x']
        y = node['y']
        if 'label' not in node:
            nodes_group.circle(x, y, NODE_RADIUS)
        else:
            classname = 'input-node' if node['layer'] == 0 else 'output-node'
            offset = -1 if node['layer'] == 0 else 1

            node_group = nodes_group.add('g', {
                'class': classname,
                'transform': f'translate({x},{y})'}
            )
            node_group.add('circle', {'r': NODE_RADIUS})
            node_group.add('text', {'class': 'activation-value'}, 0)
            node_group.add('text', {'x': offset * (5 + NODE_RADIUS)}, html.escape(node['label']))

            if node['layer'] == 0:
                rect_x = -NODE_RADIUS - MARGIN_X
                rect_width = MARGIN_X + 2 * NODE_RADIUS + 2
                node_group.rect(rect_x, -NODE_RADIUS - 1, rect_width, 2 * NODE_RADIUS + 2, classname = 'hit-box')

    max_weight = max(edge['weight'] for edge in network['edges'])
    min_weight = min(edge['weight'] for edge in network['edges'])

    for edge in network['edges']:
        node1 = nodes[edge['node1']]
        node2 = nodes[edge['node2']]
        x1 = node1['x']
        y1 = node1['y']
        x2 = node2['x']
        y2 = node2['y']
        dx = x2 - x1
        dy = y2 - y1
        d = (dx ** 2 + dy ** 2) ** 0.5
        if d > 0:
            offset_x = dx / d * (NODE_RADIUS + 2)
            offset_y = dy / d * (NODE_RADIUS + 2)
            x1 += offset_x
            y1 += offset_y
            x2 -= offset_x
            y2 -= offset_y
        
        if edge['weight'] > 0:
            colour = lerp_colour(edge['weight'], max_weight, GREY, BLUE)
        else:
            # colour = lerp_colour(-edge['weight'], -min_weight, GREY, RED)
            colour = lerp_colour(-edge['weight'], -min_weight, GREY, YELLOW)

        stroke = f"rgb({colour[0]},{colour[1]},{colour[2]})"
        edges_group.line(x1, y1, x2, y2, color=stroke)

    activations = get_activation_pattern(network, layout, False)
    print(activations)
    _add_script(svg, activations, 'network_activation.js')

    return svg


def draw_network_svg_2(svg_id, network):
    max_nodes = max(len(layer) for layer in network['layout'])
    n_layers = len(network['layout'])

    svg_width = 2 * (MARGIN_X + NODE_RADIUS) + (n_layers - 1) * NODE_DX
    svg_height = 2 * (MARGIN_Y + NODE_RADIUS) + (max_nodes - 1) * NODE_DY

    svg = SVG({'id': svg_id, 'viewBox': f"0 0 {svg_width} {svg_height}"})
    _add_styles(svg)

    svg.rect(0, 0, svg_width, svg_height, classname='hit-box background')

    nodes_group = svg.add('g', {'class': 'node'})
    edges_group = svg.add('g', {'class': 'edge'})

    max_weight = max(edge['weight'] for edge in network['edges'])
    min_weight = min(edge['weight'] for edge in network['edges'])

    # Position nodes in the SVG based on their layer and index within the layer
    for layer_i, layer in enumerate(network['layout']):
        x = MARGIN_X + layer_i * NODE_DX + NODE_RADIUS

        for node_id in layer:
            node = network['nodes'][node_id]
            y = MARGIN_Y + (max_nodes - len(layer)) * NODE_DY / 2 + layer.index(node_id) * NODE_DY + NODE_RADIUS
            node['x'] = x
            node['y'] = y

            if 'label' not in node:
                nodes_group.circle(x, y, NODE_RADIUS)
            else:
                classname = 'input-node' if layer_i == 0 else 'output-node'
                offset = -1 if layer_i == 0 else 1

                node_group = nodes_group.add('g', {
                    'class': classname,
                    'transform': f'translate({x},{y})'}
                )
                node_group.add('circle', {'r': NODE_RADIUS})
                node_group.add('text', {'class': 'activation-value'}, 0)
                node_group.add('text', {'x': offset * (5 + NODE_RADIUS)}, html.escape(node['label']))

                if layer_i == 0:
                    rect_x = -NODE_RADIUS - MARGIN_X
                    rect_width = MARGIN_X + 2 * NODE_RADIUS + 2
                    node_group.rect(rect_x, -NODE_RADIUS - 1, rect_width, 2 * NODE_RADIUS + 2, classname = 'hit-box')

    # Draw edges between nodes based on the network's edges
    for edge in network['edges']:
        node1 = network['nodes'][edge['node1']]
        node2 = network['nodes'][edge['node2']]
        x1 = node1['x']
        y1 = node1['y']
        x2 = node2['x']
        y2 = node2['y']
        dx = x2 - x1
        dy = y2 - y1
        d = (dx ** 2 + dy ** 2) ** 0.5
        if d > 0:
            offset_x = dx / d * (NODE_RADIUS + 2)
            offset_y = dy / d * (NODE_RADIUS + 2)
            x1 += offset_x
            y1 += offset_y
            x2 -= offset_x
            y2 -= offset_y
        
        if edge['weight'] > 0:
            colour = lerp_colour(edge['weight'], max_weight, GREY, BLUE)
        else:
            colour = lerp_colour(-edge['weight'], -min_weight, GREY, YELLOW)

        stroke = f"rgb({colour[0]},{colour[1]},{colour[2]})"
        edges_group.line(x1, y1, x2, y2, color=stroke)

    return svg


def draw_chain(svg_id, layers, transitions):
    MARGIN = 10 + NODE_RADIUS
    NODE_DX = 200
    NODE_DY = 40 + NODE_RADIUS * 2

    nodes_x = len(layers)
    nodes_y = max(len(layer) for layer in layers)
    width = 2 * MARGIN + (nodes_x - 1) * NODE_DX
    height = 2 * MARGIN + (nodes_y - 1) * NODE_DY
    mid_y = height / 2

    svg = SVG({'id': svg_id, 'viewBox': f'0 0 {width} {height}'})
    _add_styles(svg)
    _add_arrow_marker(svg)

    node_g = svg.add('g', {'class': 'node'})
    node_positions = {}

    for i, layer in enumerate(layers):
        x = MARGIN + i * NODE_DX
        nodes_in_layer = len(layer)
        layer_height = (nodes_in_layer - 1) * NODE_DY
        layer_start_y = mid_y - layer_height / 2
        for j, node in enumerate(layer):
            y = layer_start_y + j * NODE_DY
            node_positions[node] = (x, y )
            node_g.circle(x, y, 20)
            node_g.add('text', {'x': x, 'y': y + NODE_RADIUS + 10, 'text-anchor': 'middle','class': 'plot-label'}, html.escape(node))

    edge_g = svg.add('g', {'class': 'edge-arrow'})

    for transition in transitions:
        from_node, to_node = transition
        x1, y1 = node_positions[from_node]
        x2, y2 = node_positions[to_node]
        x1, y1, x2, y2 = offset_line(x1, y1, x2, y2, NODE_RADIUS + 2, NODE_RADIUS + 6)
        edge_g.line(x1, y1, x2, y2)

    svg.write(f'{svg_id}.svg')


def output_map(folder, svg_id):
    """ Draw a 2D axis as a heatmap of what the output token will be. """

    filename = os.path.join(folder, "model_output.txt")
    data = parse_model_data(filename)
    tokens = data['tokens']
    weights = data['weights'][-2]
    biases = [d[0] for d in data['weights'][-1]]
    print(tokens)
    print(weights)
    print(biases)

    RANGE = 4   # Values in the input space will be between -RANGE and RANGE
    SIZE = 100  # Size of the SVG canvas in each direction
    STEP = 2
    # COLOURS = [
    #     'rgb(255, 255, 255)',   # BR = white (not actually drawn)
    #     'rgb(0, 60, 128)',      # are = blue
    #     'rgb(0, 255, 255)',     # herbivores = green
    #     'rgb(255, 0, 175)',     # sheep = pink
    #     'rgb(255, 255, 0)'      # slow = yellow
    # ]

    svg = SVG({'id': svg_id, 'viewBox': f'{-SIZE} {-SIZE} {SIZE * 2} {SIZE * 2}'})
    n = len(tokens)

    for x in range(-SIZE, SIZE + 1, STEP):
        for y in range(-SIZE, SIZE + 1, STEP):
            input_vector = [x * RANGE / SIZE, y * RANGE / SIZE]
            output_vector = [sum(input_vector[i] * weights[j][i] + biases[j] for i in range(2)) for j in range(n)]
            exp_vector = [pow(2.71828, v) for v in output_vector]
            sum_exp = sum(exp_vector)
            if sum_exp > 0:
                output_vector = [v / sum_exp for v in exp_vector]

            # Get weighted average of output_vector using RGB_COLORS as weights
            red = int(sum(output_vector[i] * RGB_COLORS[i][0] for i in range(n)))
            green = int(sum(output_vector[i] * RGB_COLORS[i][1] for i in range(n)))
            blue = int(sum(output_vector[i] * RGB_COLORS[i][2] for i in range(n)))
            colour = f'rgb({red}, {green}, {blue})'

            svg.rect(x - STEP / 2 - 0.5, - y - STEP / 2 - 0.5, STEP + 1, STEP + 1, fill=colour)

            # max_output = max(output_vector)
            # max_index = output_vector.index(max_output)
            # if max_index > 0:
            #     colour = COLOURS[max_index]
            #     svg.rect(x - STEP / 2 - 0.5, - y - STEP / 2 - 0.5, STEP + 1, STEP + 1, fill=colour)

    # Axis
    svg.add('line', {'x1': -SIZE, 'y1': 0, 'x2': SIZE, 'y2': 0, 'stroke': 'black'})
    svg.add('line', {'x1': 0, 'y1': -SIZE, 'x2': 0, 'y2': SIZE, 'stroke': 'black'})
    svg.write(f'{svg_id}.svg')


def output_map_with_lines(folder, svg_id):
    """ Draw a 2D axis as a heatmap of what the output token will be. """

    filename = os.path.join(folder, "model_output.txt")
    data = parse_model_data(filename)
    tokens = data['tokens']
    embeddings = data['weights'][0]
    weights = data['weights'][-2]
    biases = [d[0] for d in data['weights'][-1]]

    print(tokens)
    print(weights)
    print(biases)
    print(embeddings)

    max_x = max(abs(embedding[0]) for embedding in embeddings)
    max_y = max(abs(embedding[1]) for embedding in embeddings)
    max_v = max(max_x, max_y)
    print(max_v)
    SIZE = 100
    SCALE = SIZE / max_v if max_v != 0 else 1

    comparisons = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            dx = weights[i][0] - weights[j][0]
            dy = weights[i][1] - weights[j][1]
            dz = biases[i] - biases[j]

            if abs(dx) > abs(dy):
                y1 = -max_v * SCALE - 1
                y2 = max_v * SCALE + 1
                x1 = (y1 * dy + dz * SCALE) / -dx
                x2 = (y2 * dy + dz * SCALE) / -dx
            else:
                x1 = -max_v * SCALE - 1
                x2 = max_v * SCALE + 1
                y1 = (x1 * dx + dz * SCALE) / -dy
                y2 = (x2 * dx + dz * SCALE) / -dy

            # Determine which side of the line the first token is preferred.
            side = 1 if dx * dy > 0 else -1

            comparisons.append([i, j, x1, y1, x2, y2, side])

    svg = SVG({'id': svg_id, 'viewBox': f'{-SIZE} {-SIZE} {SIZE * 2} {SIZE * 2}'})

    # for i in range(len(tokens)):
    #     area = find_surrounded_area(comparisons, i, SIZE)
    #     colour = f'rgb({RGB_COLORS[i][0]},{RGB_COLORS[i][1]},{RGB_COLORS[i][2]})'
    #     svg.add('polygon', {'points': ' '.join(f'{x},{-y}' for x, y in area), 'fill': colour, 'opacity': 0.25})

    area = find_surrounded_area(comparisons, 0, SIZE)
    colour = f'rgb({RGB_COLORS[1][0]},{RGB_COLORS[1][1]},{RGB_COLORS[1][2]})'
    svg.add('polygon', {'points': ' '.join(f'{x},{-y}' for x, y in area), 'fill': colour, 'opacity': 0.25})

    for comparison in comparisons:
        i, j, x1, y1, x2, y2, side = comparison
        if i == 1 or j == 1:
            svg.add('line', {'x1': x1, 'y1': -y1, 'x2': x2, 'y2': -y2, 'stroke': 'black', 'opacity': 0.25})

    # Axis
    svg.add('line', {'x1': -SIZE, 'y1': 0, 'x2': SIZE, 'y2': 0, 'stroke': 'black'})
    svg.add('line', {'x1': 0, 'y1': -SIZE, 'x2': 0, 'y2': SIZE, 'stroke': 'black'})

    output_file = os.path.join(folder, f'{svg_id}.svg')
    svg.write(output_file)


def find_surrounded_area(comparisons, token, size):
    filtered_comparisons = [c for c in comparisons if c[0] == token or c[1] == token]

    # Points of the bounding box
    bounding_points = [[size, size], [size, -size], [-size, -size], [-size, size]]

    for comparison in filtered_comparisons:
        print("\nComparison:", comparison)
        i, j, x1, y1, x2, y2, side = comparison
        # Determine where the comparison line intersects the bounding box
        intersections = []
        for p1, bounding_point1 in enumerate(bounding_points):
            p2 = (p1 + 1) % len(bounding_points)
            bx1, by1 = bounding_point1
            bx2, by2 = bounding_points[p2]

            denom = (x2 - x1) * (by2 - by1) - (y2 - y1) * (bx2 - bx1)
            if denom != 0:
                ua = ((bx2 - bx1) * (y1 - by1) - (by2 - by1) * (x1 - bx1)) / denom
                ub = ((x2 - x1) * (y1 - by1) - (y2 - y1) * (x1 - bx1)) / denom
                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    print('intersection at ', p1, p2)
                    intersections.append((p2, x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)))

        # Update bounding box based on the intersections
        if len(intersections) != 2:
            print("Unexpected number of intersections:", len(intersections))
            break

        print("Intersections:", intersections)

        # Get the intersections in the correct order based on the side of the line that is preferred.
        dx = intersections[0][1] - intersections[1][1]
        dy = intersections[0][2] - intersections[1][2]
        
        # Swap side depending on which token is preferred in the comparison
        side = -side if (i == token) else side
        if dx * dy * side < 0:
            intersections.reverse()

        new_bounding_points = [intersections[0][1:], intersections[1][1:]]
        start_point = intersections[1][0]
        end_point = intersections[0][0]
        n = len(bounding_points)

        print("start at :", start_point, "end at:", end_point, "side:", side)

        while start_point != end_point:
            print("Adding bounding point:", start_point)
            new_bounding_points.append(bounding_points[start_point])
            start_point = (start_point + 1) % n

        bounding_points = new_bounding_points
        print("Updated bounding points:", bounding_points)

    return bounding_points


def draw_network_1(folder, svg_id):
    """
    Draw a fully connected network of nodes with two layers representing the tokens in token_list.
    """
    filename = os.path.join(folder, "model_output.txt")
    data = parse_model_data(filename)
    n = len(data['tokens'])
    layout = [n, n]

    network = get_network(layout, data['tokens'], data['weights'])
    draw_network_svg(svg_id, data['tokens'], layout, network)


def draw_network_2(folder, svg_id, suffix=None):
    """
    Draw a fully connected network of nodes with an input layer, hidden layer, and output layer.
    """

    filepath = get_filepath(folder, "model_output", suffix)
    data = parse_model_data(filepath)
    network = get_network_2(data)

    svg = draw_network_svg_2(svg_id, network)

    svg_filename = f'{svg_id}.svg' if suffix is None else f'{svg_id}_{suffix}.svg'
    svg.write(os.path.join(folder, svg_filename))


def draw_token_embeddings(folder, svg_id, suffix=None):
    AXIS = 100
    SIZE = AXIS + 15

    filepath = get_filepath(folder, "model_output", suffix)
    data = parse_model_data(filepath)
    weights = data['weights'][0]
    
    max_weight = max(abs(weight) for row in weights for weight in row)
    scale = AXIS / max_weight if max_weight != 0 else 1

    svg = SVG({'viewBox': f"{-SIZE} {-SIZE} {SIZE * 2} {SIZE * 2}", 'width': SIZE * 2, 'height': SIZE * 2})

    svg.add_style('line.axis', {'stroke': 'black', 'stroke-width': 1})
    svg.add_style('path.cross', {'stroke': 'red', 'stroke-width': 1, 'fill': 'none'})
    svg.add_style('text.label', {'font-size': '12px', 'text-anchor': 'middle'})

    svg.line(0, -AXIS, 0, AXIS, attrs={'class': 'axis'})
    svg.line(-AXIS, 0, AXIS, 0, attrs={'class': 'axis'})

    for i, row in enumerate(weights):
        x = round(row[0] * scale, 2)
        y = round(-row[1] * scale, 2)
        svg.add('path', {'d': f'M{x - 3.5} {y - 3.5} l7 7 M{x - 3.5} {y + 3.5} l7 -7', 'class': 'cross'})
        svg.add('text', {'x': x, 'y': y - 7, 'class': 'label'}, html.escape(data['tokens'][i]))

    svg_filename = f'{svg_id}.svg' if suffix is None else f'{svg_id}_{suffix}.svg'
    svg.write(os.path.join(folder, svg_filename))


def draw_chain_1():
    nodes = [['<BR>'], ['sheep'], ['are', 'eat'], ['herbivores', 'slow', 'grass'], ['<END>']]
    transitions = [
        ['<BR>', 'sheep'],
        ['sheep', 'are'],
        ['sheep', 'eat'],
        ['are', 'herbivores'],
        ['are', 'slow'],
        ['eat', 'grass'],
        ['herbivores', '<END>'],
        ['slow', '<END>'],
        ['grass', '<END>']
    ]
    draw_chain('simple_chain', nodes, transitions)


def draw_chain_2():
    nodes = [['<BR>'], ['sheep', 'rabbits'], ['are', 'eat', 'like'], ['herbivores', 'slow', 'grass', 'running'], ['<END>']]
    transitions = [
        ['<BR>', 'sheep'],
        ['<BR>', 'rabbits'],
        ['sheep', 'are'],
        ['sheep', 'eat'],
        ['rabbits', 'are'],
        ['rabbits', 'eat'],
        ['rabbits', 'like'],
        ['are', 'herbivores'],
        ['are', 'slow'],
        ['eat', 'grass'],
        ['like', 'running'],
        ['herbivores', '<END>'],
        ['slow', '<END>'],
        ['grass', '<END>'],
        ['running', '<END>']
    ]
    print(nodes)
    draw_chain('simple_chain', nodes, transitions)


if __name__ == "__main__":
    # draw_network_1("example1", 'activation-network')
    draw_network_2("example2", 'activation-network')
    # draw_token_embeddings("example2", 'token-embeddings')
    # draw_token_embeddings("example3", 'token-embeddings', "4")
    # draw_token_embeddings("example4", 'token-embeddings', "1_good")
    # draw_chain_1()
    # draw_chain_2()

    # output_map(os.path.join('example4', 'context2'), 'output-map')
    # output_map('example2', 'output-map-heatmap')
    # output_map_with_lines('example2', 'output-map-lines')
