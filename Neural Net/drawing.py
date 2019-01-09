"""
Holds methods for drawing things involving the neural_net module.
"""
import pygame

import nn


# James Lecomte
# Started     04/06/18
# Last edit   06/06/18


MAX_NODE_RADIUS = 10

def draw_neural_net(surface: pygame.surface.SurfaceType, neural_net: nn.NeuralNet, position=None, size=None):
    """Draw a neural net to a surface."""
    if position is None:
        position = (0, 0)
    if size is None:
        size = surface.get_size()

    layersizes = list(layer.input_size for layer in neural_net.layers)
    layersizes.append(neural_net.layers[-1].output_size)
    max_nodes = max(layersizes)
    layers = len(neural_net.layers)+1
    subsurface = surface.subsurface(pygame.Rect(position, size))
    width = subsurface.get_width()
    height = subsurface.get_height()

    # Makes sure nodes don't overlap and aren't too crowded
    node_radius = min(MAX_NODE_RADIUS, (height/max_nodes)*0.25, (width/layers)*0.2)
    node_radius_rounded = round(node_radius)

    # Draw edges
    n_in_x = node_radius
    for layer_num, layer in enumerate(neural_net.layers, 1):
        n_out_x = _get_layer_x(width, node_radius, layers, layer_num)
        for n_out, n_out_y in enumerate(_get_node_y_positions(height, node_radius, max_nodes, layer.output_size)):
            for n_in, n_in_y in enumerate(_get_node_y_positions(height, node_radius, max_nodes, layer.input_size)):
                weight = layer.weights[n_out, n_in]
                if weight > 0:
                    color = (80, 120+weight*100, 80)
                else:
                    color = (120-weight*100, 80, 80)
                thickness = int(abs(weight) * node_radius*1)
                if thickness >= 1:
                    pygame.draw.line(subsurface, color, (n_in_x, n_in_y), (n_out_x, n_out_y), thickness)

        n_in_x = n_out_x

    # Draw input nodes
    for y in _get_node_y_positions(height, node_radius, max_nodes, neural_net.layers[0].input_size):
        pygame.draw.circle(subsurface, (50, 50, 50), (node_radius_rounded, int(y)), node_radius_rounded)

    # Draw other nodes
    for layer_num, layer in enumerate(neural_net.layers, 1):
        n_out_x = _get_layer_x(width, node_radius, layers, layer_num)
        for n_out, n_out_y in enumerate(_get_node_y_positions(height, node_radius, max_nodes, layer.output_size)):
            bias = layer.weights[n_out, -1]
            if bias > 0:
                color = (50, int(50+bias*170), 50)
            else:
                color = (int(50+abs(bias*170)), 50, 50)
            pygame.draw.circle(subsurface, color, (int(n_out_x), int(n_out_y)), node_radius_rounded)

def _get_node_y_positions(height, node_radius, max_nodes, num_of_nodes):
    """Return the y position of nodes."""
    if max_nodes == 1:
        return tuple(height/2)

    spacing = (height-node_radius*2) / (max_nodes-1)
    top = height/2 - spacing*(num_of_nodes-1)/2
    return (top+i*spacing for i in range(num_of_nodes))

def _get_layer_x(width, node_radius, layers, layer_num):
    """Return the x position of a layer."""
    spacing = (width-node_radius*2) / (layers-1)
    return node_radius + layer_num*spacing
