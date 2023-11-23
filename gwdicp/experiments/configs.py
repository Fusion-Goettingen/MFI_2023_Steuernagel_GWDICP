"""
Contains constants and configuration for the experiments.
"""

import numpy as np

from gwdicp.utils.environment_definition import AVENUE
from gwdicp.utils.utils import tuple_to_transformation

_standard_R = np.diag([0.05, 0.05, 0.01])
_standard_environment = np.array(AVENUE)
_environment_center = np.array([0, 0, 0])  # ensure this matches your env

CONFIG_DEFAULT = {
    "variation_parameter": None,
    "n_monte_carlo_runs_per_setting": 500,

    "offset_centers": np.array([3, 0, 0]),
    "gt_transform": tuple_to_transformation(np.array([0.5, 0, 0]), np.array([0, np.pi / 16, 0])),
    "sensor_range": 16,
    "environment": _standard_environment,
    "center_point": _environment_center,
    "sample_density": 5,
    "n_layers": 8,
    "max_height": 5,
    "meas_cov": _standard_R,
}
