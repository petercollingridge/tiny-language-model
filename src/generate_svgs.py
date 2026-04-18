import html
import os
from vis.draw_svg import SVG

MARGIN_X = 85
MARGIN_Y = 10
NODE_RADIUS = 20
NODE_DX = 200
NODE_DY = 15

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


def get_network(layout, tokens, weights):
    """
    Create a network layout of nodes and edges.
    """

    nodes = []
    edges = []
    x = MARGIN_X + NODE_RADIUS

    for layer_n, layer in enumerate(layout):
        for i in range(layer):
            y = MARGIN_Y + NODE_RADIUS + i * (2 * NODE_RADIUS + NODE_DY)
            node = { 'x': x, 'y': y, 'layer': layer_n }
            if layer_n == 0 or layer_n == len(layout) - 1:
                node['label'] = tokens[i]
            nodes.append(node)
        x += NODE_DX + 2 * NODE_RADIUS

    for layer_weights in weights:
        for i, row in enumerate(layer_weights):
            n = len(row)
            for j, weight in enumerate(row):
                edge = { 'node1': nodes[i], 'node2': nodes[n + j], 'weight': weight }
                edges.append(edge)

    return { 'nodes': nodes, 'edges': edges, 'layout': layout }


def add_styles(svg):
    svg.add_style('.node circle', {'fill': 'none', 'stroke': '#111', 'stroke-width': 1})
    svg.add_style('.node text', {'dominant-baseline': 'middle'})
    svg.add_style('.input-node', {'text-anchor': 'end'})
    svg.add_style('.output-node', {'text-anchor': 'start'})
    svg.add_style('.edge line', {'stroke': '#aaa', 'stroke-width': 2, 'opacity': 0.8})


def draw_network_svg(token_list, network):
    """ Draw a fully connected network of nodes representing the tokens in token_list. """

    n_tokens = len(token_list)

    svg_width = 2 * (MARGIN_X + NODE_RADIUS * 2) + NODE_DX
    svg_height = 2 * MARGIN_Y + n_tokens * (2 * NODE_RADIUS + NODE_DY) - NODE_DY

    svg = SVG({'viewBox': f"0 0 {svg_width} {svg_height}"})
    add_styles(svg)

    svg.rect(0, 0, svg_width, svg_height, fill='#f8f8f8')

    nodes_group = svg.add('g', {'class': 'node'})
    edges_group = svg.add('g', {'class': 'edge'})

    for node in network['nodes']:
        x = node['x']
        y = node['y']
        if 'label' not in node:
            nodes_group.circle(x, y, NODE_RADIUS)
        else:
            classname = 'input-node' if node['layer'] == 0 else 'output-node'
            label_x = x - 5 - NODE_RADIUS if node['layer'] == 0 else x + 5 + NODE_RADIUS
            node_group = nodes_group.add('g', {'class': classname})
            node_group.circle(x, y, NODE_RADIUS)
            node_group.add('text', {'x': label_x, 'y': y}, html.escape(node['label']))

    for edge in network['edges']:
        x1 = edge['node1']['x']
        y1 = edge['node1']['y']
        x2 = edge['node2']['x']
        y2 = edge['node2']['y']
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
        edges_group.line(x1, y1, x2, y2, style=f"opacity: {edge['weight']:.2f}")

    svg.write('network.svg')


def draw_network_1(folder):
    filename = os.path.join(folder, "model_output.txt")
    data = parse_data(filename)
    n = len(data['tokens'])
    layout = [n, n]

    network = get_network(layout, data['tokens'], data['weights'])
    draw_network_svg(data['tokens'], network)

    print(network)


if __name__ == "__main__":
    draw_network_1("example1")
