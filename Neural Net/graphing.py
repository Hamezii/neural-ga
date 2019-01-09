"""
Holds methods for graphing in pygame.
"""
from math import ceil

import pygame

# James Lecomte
# Started     04/07/18
# Last edit   07/07/18

def draw_bar_graph(surface: pygame.surface.SurfaceType, colors, values, rect=None):
    """Plot a bar graph using an array of values."""
    if rect is None:
        rect = surface.get_rect()

    surface.fill(colors[0], rect)
    rect = rect.inflate(-40, -40)

    max_x = max(len(values), 1)
    x_spacing = rect.width / max_x

    max_value = max(ceil(max(values)), 1)

    for i, value in enumerate(values):
        bar_height = int(rect.height * value / max_value)
        bar_left, bar_top = (rect.left + int(x_spacing*i), rect.bottom - bar_height)
        pygame.draw.rect(surface, colors[1], (bar_left, bar_top, ceil(x_spacing), bar_height))

    draw_axis(surface, colors[2], rect, max_x, max_value)

def draw_line_graph(surface: pygame.surface.SurfaceType, colors, *datasets, rect=None):
    """Plot a line graph using an array of values."""
    if rect is None:
        rect = surface.get_rect()

    surface.fill(colors[0], rect)
    rect = rect.inflate(-40, -40)

    max_len = max(len(values) for values in datasets)
    max_x = max(max_len - 1, 1)
    x_spacing = rect.width / max_x

    max_value = max(ceil(max(max(values)for values in datasets)), 1)

    for values in datasets:
        for i, value in enumerate(values):
            point_height = int(rect.height * value / max_value)
            if i: # Not first point
                old_point = point
            point = (rect.left + int(x_spacing*i), rect.bottom - point_height)
            if i:
                pygame.draw.line(surface, colors[1], old_point, point, 3)
            pygame.draw.circle(surface, colors[1], point, 3)

    draw_axis(surface, colors[2], rect, max_x, max_value)


def draw_axis(surface: pygame.surface.SurfaceType, color, rect, max_x, max_y):
    """Draw the axis for a graph on a surface."""


    # Drawing x axis and notches
    pygame.draw.line(surface, color, (rect.bottomleft), (rect.bottomright), 5)
    x_spacing = rect.width / max_x
    for i in range(max_x):
        line_x = rect.left+x_spacing*i
        pygame.draw.line(surface, color, (line_x, rect.bottom), (line_x, rect.bottom+6), 3)
    pygame.draw.line(surface, color, (rect.right-1, rect.bottom), (rect.right-1, rect.bottom+5), 3)


    # Drawing y axis and notches
    pygame.draw.line(surface, color, (rect.bottomleft), (rect.topleft), 5)
    y_spacing = rect.height / max_y
    for i in range(max_y):
        line_y = rect.bottom-int(y_spacing * i)
        pygame.draw.line(surface, color, (rect.left, line_y), (rect.left-5, line_y), 3)

    pygame.draw.line(surface, color, (rect.left, rect.top+1), (rect.left-6, rect.top+1), 3)
