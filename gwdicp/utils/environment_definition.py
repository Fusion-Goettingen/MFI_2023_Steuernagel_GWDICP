"""
Contains the definition of the simulation environment
"""
import numpy as np


def get_aligned_square(center, half_axis_size):
    center = np.array(center)
    return [[center[0] - half_axis_size, center[1] - half_axis_size, center[0] + half_axis_size,
             center[1] - half_axis_size],
            [center[0] + half_axis_size, center[1] - half_axis_size, center[0] + half_axis_size,
             center[1] + half_axis_size],
            [center[0] + half_axis_size, center[1] + half_axis_size, center[0] - half_axis_size,
             center[1] + half_axis_size],
            [center[0] - half_axis_size, center[1] + half_axis_size, center[0] - half_axis_size,
             center[1] - half_axis_size]]


AVENUE = np.array([
    # [0, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 0, 0]
    *get_aligned_square([0, 2], 0.3),
    *get_aligned_square([0, -2], 0.32),
    *get_aligned_square([3, 2], 0.3),
    *get_aligned_square([3.5, -2], 0.35),
    *get_aligned_square([-2.7, 2], 0.25),
    *get_aligned_square([-3.2, -2], 0.31),
    [6, 1.8, 14, 1.8],
    [6, 1.8, 6, 5],
    *get_aligned_square([7, -2], 0.3),
    *get_aligned_square([10.2, -2.05], 0.3),
    *get_aligned_square([13, -1.9], 0.3),
    [4, -3.3, 4, -7], [7, -3.3, 7, -7],
    [7, -3.3, 13, -3.3]
])
