import os
import html

from draw_svg import SVG


NODE_R = 20
MARGIN = 10


def draw_network_svg(token_list):
    """ Draw a fully connected network of nodes representing the tokens in token_list. """

    n = len(token_list)
    node_gap_x = 200
    node_gap_y = 15
    label_x = 80

    svg_width = 2 * (MARGIN * 2 + NODE_R * 2 + label_x) + node_gap_x
    svg_height = 2 * MARGIN + n * (2 * NODE_R + node_gap_y) - node_gap_y

    svg = SVG({'viewBox': f"0 0 {svg_width} {svg_height}"})
    svg.add_style('.node', {'fill': '#aaa'})
    svg.add_style('.node-label', {'dominant-baseline': 'middle'})
    svg.add_style('.edge', {'stroke': '#aaa', 'stroke-width': 2, 'opacity': 0.8})
    svg.add_style('.label-left', {'text-anchor': 'end'})

    # svg.add('rect', {'x': 0, 'y': 0, 'width': svg_width, 'height': svg_height, 'fill': '#eee', 'stroke': '#ccc'})

    x1 = 2 * MARGIN + label_x + NODE_R
    x2 = x1 + node_gap_x + NODE_R * 2
    y_coords = []
    for i, token in enumerate(token_list):
        y = MARGIN + i * (2 * NODE_R + node_gap_y) + NODE_R
        y_coords.append(y)
        svg.circle(x1, y, NODE_R, classname='node')
        svg.circle(x2, y, NODE_R, classname='node')
        svg.add('text', {'x': x1 - MARGIN - NODE_R, 'y': y, 'class': 'node-label label-left'}, html.escape(token))
        svg.add('text', {'x': x2 + MARGIN + NODE_R, 'y': y, 'class': 'node-label'}, html.escape(token))

    dx = x2 - x1
    for i in range(n):
        for j in range(n):
            dy = y_coords[j] - y_coords[i]
            d = (dx ** 2 + dy ** 2) ** 0.5
            dx1 = dx * (NODE_R + 2) / d
            dy1 = dy * (NODE_R + 2) / d
            svg.add('line', {'x1': x1 + dx1, 'y1': y_coords[i] + dy1, 'x2': x2 - dx1, 'y2': y_coords[j] - dy1, 'class': 'edge'})

    svg.write('network.svg')

def draw_markov_chain_1(token_list):
    n = len(token_list)
    edge_length = 100 + NODE_R * 2
    svg_width = MARGIN * 2 + NODE_R * 2 + edge_length * (n - 1)
    svg_height = MARGIN * 2 + NODE_R * 2 * 3 + 20

    svg = SVG({'viewBox': f"0 0 {svg_width} {svg_height}"})
    svg.add_style('.node', {'fill': 'none', 'stroke': '#aaa', 'stroke-width': 1})
    svg.add_style('.node-label', {'dominant-baseline': 'hanging', 'text-anchor': 'middle'})
    svg.add_style('.edge', {'stroke': '#aaa', 'stroke-width': 2, 'opacity': 0.8, 'marker-end': 'url(#arrow)'})

    defs_element = svg.add('defs')
    marker_element = defs_element.add('marker', {'id': 'arrow', 'viewBox': '0 0 10 10', 'refX': '5', 'refY': '5', 'orient': 'auto-start-reverse'})
    marker_element.add('path', {'d': 'M0 0L10 5L0 10z', 'fill': '#aaa'})

    svg.rect(0, 0, svg_width, svg_height, fill="#eee", stroke="#ccc")

    x = MARGIN + NODE_R
    y = MARGIN + NODE_R * 3
    svg.circle(x, y, NODE_R, classname='node')
    svg.add('text', {'x': x, 'y': y + NODE_R + 5, 'class': 'node-label'}, html.escape(token_list[0]))

    svg.line(x + NODE_R + 2, y, x + edge_length - NODE_R - 4, y, attrs={'class': 'edge'})

    x += edge_length
    svg.circle(x, y, NODE_R, classname='node')
    svg.add('text', {'x': x, 'y': y + NODE_R + 5, 'class': 'node-label'}, html.escape(token_list[3]))

    svg.line(x + NODE_R + 2, y, x + edge_length - NODE_R - 4, y, attrs={'class': 'edge'})

    x += edge_length
    svg.circle(x, y, NODE_R, classname='node')
    svg.add('text', {'x': x, 'y': y + NODE_R + 5, 'class': 'node-label'}, html.escape(token_list[1]))

    dx = edge_length
    dy = NODE_R * 2
    d = (dx ** 2 + dy ** 2) ** 0.5
    dx1 = dx * (NODE_R + 2) / d
    dy1 = dy * (NODE_R + 2) / d

    dx2 = dx * (NODE_R + 5) / d
    dy2 = dy * (NODE_R + 5) / d

    svg.line(x + dx1, y - dy1, x + edge_length - dx2, y - NODE_R * 2 + dy2, attrs={'class': 'edge'})
    svg.line(x + dx1, y + dy1, x + edge_length - dx2, y + NODE_R * 2 - dy2, attrs={'class': 'edge'})

    x += edge_length
    svg.circle(x, y - NODE_R * 2, NODE_R, classname='node')
    svg.add('text', {'x': x, 'y': y - NODE_R + 5, 'class': 'node-label'}, html.escape(token_list[4]))

    svg.circle(x, y + NODE_R * 2, NODE_R, classname='node')
    svg.add('text', {'x': x, 'y': y + NODE_R * 3 + 5, 'class': 'node-label'}, html.escape(token_list[2]))

    svg.line(x + dx1, y - NODE_R * 2 + dy1, x + edge_length - dx2, y - dy2, attrs={'class': 'edge'})
    svg.line(x + dx1, y + NODE_R * 2 - dy1, x + edge_length - dx2, y + dy2, attrs={'class': 'edge'})

    x += edge_length
    svg.circle(x, y, NODE_R, classname='node')
    svg.add('text', {'x': x, 'y': y + NODE_R + 5, 'class': 'node-label'}, html.escape(token_list[0]))

    svg.write('markov_chain_1.svg')


if __name__ == '__main__':
    token_list = ['<BR>', 'are', 'herbivores', 'sheep', 'slow']
    # draw_network_svg(token_list)

    draw_markov_chain_1(token_list)
