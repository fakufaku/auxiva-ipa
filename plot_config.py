import colorsys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from colorspacious import cspace_convert, cspace_converter


def color_palette(n_colors=6, sat=0.5, v_scale="sqrt"):

    h = np.linspace(0, 1, n_colors, endpoint=False)[::-1]
    s = [sat] * n_colors

    if v_scale == "linear":
        v = np.linspace(0, 1, n_colors)
    elif v_scale == "log":
        v = np.log10(np.linspace(1, 10, n_colors))
    elif v_scale == "sqrt":
        v = np.sqrt(np.linspace(0, 1, n_colors))
    else:
        raise NotImplementedError()

    return [colorsys.hsv_to_rgb(*t) for t in zip(h, s, v)]


def color_palette2(
    n_colors=6, chroma=0.5, hue_offset=0.1, v_scale="linear", max_lum=1.0
):

    h = 360 * hue_offset + np.linspace(0, 360, n_colors, endpoint=False)[::-1]
    C = [chroma * 100] * n_colors

    if v_scale == "linear":
        J = np.linspace(0, max_lum, n_colors)
    elif v_scale == "log":
        J = np.log10(np.linspace(1, 10 ** max_lum, n_colors))
    elif v_scale == "sqrt":
        J = np.sqrt(np.linspace(0, max_lum, n_colors))
    else:
        raise NotImplementedError()
    J *= 100
    J = np.maximum(1e-1, J)

    # convert to RGB
    pal = cspace_convert(list(zip(J, C, h)), "JCh", "sRGB1")

    # clip RGB values to [0, 1]
    pal = np.maximum(0, pal)
    pal = np.minimum(1, pal)

    return pal


def seaborn_config(n_colors, style="white", font="sans-serif", font_scale=0.75):
    sns.set_theme(context="paper", style=style, font=font, font_scale=font_scale)

    # pal = sns.color_palette("cubehelix", n_colors=n_colors)
    # sns.set_palette("viridis", n_colors=7)
    # sns.set_palette(pal, n_colors=n_colors)

    pal = color_palette2(n_colors=n_colors, hue_offset=0.11, chroma=0.5, max_lum=0.8)

    # colorblind friendly color palette
    # https://gist.github.com/thriveth/8560036
    if n_colors == 6:
        pal = [
            "#6494aa",
            "#a63d40",
            "#90a959",
            "#e9b872",
            "#6CA9B2",
            "#333333",
        ]
    elif n_colors == 7:
        pal = [
            "#6494aa",
            "#a63d40",
            "#90a959",
            "#e9b872",
            "#6CA9B2",
            "#333333",
            "#a9ddd6",
        ]
    else:
        raise NotImplementedError

    sns.set_palette(pal)


if __name__ == "__main__":

    for o in np.linspace(0, 1, 11):
        pal = color_palette2(n_colors=6, hue_offset=0.11, max_lum=0.8)
        sns.palplot(pal)
        plt.title(o)
        plt.tight_layout()
    plt.show()
