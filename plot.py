"""
Some reusable functions for plotting.
"""
from __future__ import print_function, division
from matplotlib.pyplot import cm
import numpy as np


def set_font_size(ax, font_size):
    """
    Set font size of all axis text objects to specified value.
    """

    for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        txt.set_fontsize(font_size)


def set_colors(ax, color):
    """Set colors on all parts of axis."""

    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color)
    ax.spines['left'].set_color(color)
    ax.spines['right'].set_color(color)

    ax.tick_params(axis='x', color=color)
    ax.tick_params(axis='y', color=color)

    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color(color)

    ax.title.set_color(color)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)


def get_n_colors(n, colormap='rainbow'):
    """
    Return a list of colors equally spaced over a color map.
    :param n: number of colors
    :param colormap: colormap to use
    :return: list of colors that can be passed directly to color argument of plotting
    """

    return getattr(cm, colormap)(np.linspace(0, 1, n))
