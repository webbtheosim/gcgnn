import colorsys
import numpy as np


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(*rgb_color)


def adjust_saturation(hex_color, x):
    rgb_color = hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*[c / 255.0 for c in rgb_color])
    s *= x / 100
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return rgb_to_hex((int(r * 255), int(g * 255), int(b * 255)))


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip("#")

    red = int(hex_color[0:2], 16) / 255.0
    green = int(hex_color[2:4], 16) / 255.0
    blue = int(hex_color[4:6], 16) / 255.0
    alpha = int(hex_color[6:8], 16) / 255.0 if len(hex_color) >= 8 else 1.0

    return (red, green, blue, alpha)
