"""
Minimal script to quickly visualize simulated environment data used in the simulation study.
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from data_generation import draw_multilayer_samples_from_env
from gwdicp.experiments.configs import CONFIG_DEFAULT


def main():
    seed = 42  # seed for RNG
    config = deepcopy(CONFIG_DEFAULT)

    rng = np.random.default_rng(seed)

    single_layer = draw_multilayer_samples_from_env(
        rng=rng,
        line_segment_list=config["environment"],
        sample_density=config["sample_density"],
        n_layers=1,
        dist_between_layers=0
    )

    plt.scatter(single_layer[:, 0], single_layer[:, 1], marker='.')
    plt.axis('equal')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()


if __name__ == '__main__':
    main()
