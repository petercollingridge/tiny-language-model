import html
import os

from vis.draw_svg import SVG

MARGIN_X = 100
MARGIN_Y = 10
NODE_RADIUS = 20
NODE_DX = 220
NODE_DY = 55

BLUE = [0, 60, 90]
GREY = [200, 200, 200]
YELLOW = [255, 180, 0]


def parse_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.splitlines()
    tokens_line = lines[1]
    tokens = tokens_line.split(':')[1].strip().split('|')

    weights = []
    matrix = None
    for line in lines[2:]:
        if line.strip():
            if line.startswith("weights:"):
                if matrix is not None:
                    weights.append(matrix)
                matrix = []
            else:
                weight = [float(x) for x in line.strip().split(",")]
                matrix.append(weight)

    if matrix is not None:
        weights.append(matrix)

    return { 'tokens': tokens, 'weights': weights }


def get_filepath(folder, filename, suffix=None):
    if suffix is not None:
        filename = f"{filename}_{suffix}"
    return os.path.join(folder, filename + ".txt")


def get_network(layout, tokens, weights):
    """
    Create a network layout of nodes and edges.
    """

    nodes = []
    edges = []
    x = MARGIN_X + NODE_RADIUS
    max_layer_size = max(layout)

    for layer_n, layer in enumerate(layout):
        start_y = MARGIN_Y + NODE_RADIUS + (max_layer_size - layer) * NODE_DY / 2
        for i in range(layer):
            y = start_y + i * NODE_DY
            node = { 'x': x, 'y': y, 'layer': layer_n, 'id': len(nodes) }
            if layer_n == 0 or layer_n == len(layout) - 1:
                node['label'] = tokens[i]
            nodes.append(node)
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


def get_activation_pattern(network, layout, softmax=True):
    """
    Create a pattern of active nodes and edges based on the weights in the network.
    """
    nodes = network['nodes']
    edges = network['edges']
    activation_patterns = []

    # For each initial input token, calculate which nodes and edges are activated based on the weights.
    for token in range(layout[0]):
        node_activations = [0] * len(nodes)
        edge_activations = [0] * len(edges)
        node_activations[token] = 1

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
    _add_script(svg, activations, 'network_activation.js')

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


def draw_network_1(folder, svg_id):
    """
    Draw a fully connected network of nodes with two layers representing the tokens in token_list.
    """
    filename = os.path.join(folder, "model_output.txt")
    data = parse_data(filename)
    n = len(data['tokens'])
    layout = [n, n]

    network = get_network(layout, data['tokens'], data['weights'])
    draw_network_svg(svg_id, data['tokens'], layout, network)

    print(network)


def draw_network_2(folder, svg_id, suffix=None):
    """
    Draw a fully connected network of nodes with an input layer, hidden layer, and output layer.
    """

    filepath = get_filepath(folder, "model_output", suffix)
    data = parse_data(filepath)
    weights = data['weights']

    n = len(weights[0])
    hidden = len(weights[0][0])
    layout = [n, hidden, n]

    network = get_network(layout, data['tokens'], data['weights'])

    svg_filename = f'{svg_id}.svg' if suffix is None else f'{svg_id}_{suffix}.svg'
    svg = draw_network_svg(svg_id, data['tokens'], layout, network)
    svg.write(os.path.join(folder, svg_filename))

    # print(network)


def draw_token_embeddings(folder, svg_id, suffix=None):
    AXIS = 100
    SIZE = AXIS + 15

    filepath = get_filepath(folder, "model_output", suffix)
    data = parse_data(filepath)
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
        y = round(row[1] * scale, 2)
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
    # draw_chain_1()
    # draw_chain_2()
