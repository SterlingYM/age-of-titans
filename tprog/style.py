"""Matplotlib style used for all paper figures."""
import matplotlib.pyplot as plt


def use_paper_style():
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        'axes.edgecolor': '0.0',
        'font.family': 'serif',
        'font.size': 12,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.2,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.major.width': 1.5,
        'xtick.minor.width': 1,
        'xtick.major.size': 3,
        'xtick.minor.size': 9 / 4,
        'ytick.major.width': 1.5,
        'ytick.minor.width': 1,
        'ytick.major.size': 3,
        'ytick.minor.size': 9 / 4,
    })
