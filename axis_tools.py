"""
Tools for futzing with axis objects.
"""
import matplotlib.pyplot as plt


def set_fontsize(ax, fontsize, legend_fontsize=None):
    """Set fontsize of all axis text objects to specified value."""

    for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        txt.set_fontsize(fontsize)

    try:

        for txt in [ax.zaxis.label] + ax.get_zticklabels():

            txt.set_fontsize(fontsize)

    except:

        pass

    if legend_fontsize is None:

        legend_fontsize = fontsize

    legend = ax.get_legend()

    if legend:

        for txt in legend.get_texts():

            txt.set_fontsize(legend_fontsize)


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
