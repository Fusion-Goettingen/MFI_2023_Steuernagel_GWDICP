"""
Analysis of the GWD, Frobenius approximation and the optimization criterion over iterations.
"""
import numpy as np
import matplotlib.pyplot as plt
from gwdicp.algorithms.gwd_icp_implementation import GWDICP
from gwdicp.utils.data_generation import draw_multilayer_samples_from_env, add_noise
from gwdicp.utils.utils import tuple_to_transformation, transform_cloud
from gwdicp.utils.environment_definition import AVENUE


def get_example_pts(rng):
    env_lines = np.array(AVENUE)
    sample_density = 5

    n_layers = 8
    max_height = 5  # height of the highest scan
    vertical_distance = max_height / n_layers

    meas_cov = np.diag([0.1, 0.1, 0.025])

    original_samples = draw_multilayer_samples_from_env(
        rng=rng,
        line_segment_list=env_lines,
        sample_density=sample_density,
        n_layers=n_layers,
        dist_between_layers=vertical_distance
    )

    noisy_samples = add_noise(rng, original_samples, meas_cov)
    return original_samples, noisy_samples


def create_plot_over_iterations():
    """
    Visualize the gwd/frob. approx/optim. criterion over iterations for a single example run
    """
    offset_centers = np.array([3, 0, 0])  # offset between centers of the two simulated scans (in addition to noise)
    sensor_radius = 12  # sensor "vision" radius in m

    rng = np.random.default_rng()
    original_samples, noisy_samples = get_example_pts(rng)
    center_point = np.array([0, 0, 0])

    icp = GWDICP(correspondence_threshold=10, max_iter=100, keypoint_dist=1, radius=0.5, th=1.11, shape_weight=1)

    init_guess = np.zeros((4, 4))
    init_guess[3, 3] = 1
    init_guess[:3, :3] = np.eye(3)

    translation = np.array([0.5, 0, 0])
    rotation = np.array([0, np.pi / 16, 0])
    error_tf = tuple_to_transformation(translation, rotation)

    center_point_shifted = offset_centers + translation

    shifted_pts = transform_cloud(get_example_pts(rng)[1] - center_point, error_tf)

    noisy_samples = noisy_samples - center_point

    noisy_samples = noisy_samples[np.linalg.norm(noisy_samples - center_point, axis=1) < sensor_radius]
    shifted_pts = shifted_pts[np.linalg.norm(shifted_pts - center_point_shifted, axis=1) < sensor_radius]

    _ = icp.register(noisy_samples, shifted_pts, init_guess)

    n_to_plot = 50
    plt.style.use("../../data/stylesheets/paper.mplstyle")
    info_dict = icp.get_information_about_last_iteration()
    mean_gwd = info_dict["mean_gwd"][:n_to_plot]
    mean_frob = info_dict["mean_frob"][:n_to_plot]
    mean_optim = info_dict["mean_optim"][:n_to_plot]

    markersize = 12
    plt.plot(mean_gwd / mean_gwd.max(), label='Gaussian Wasserstein Distance',
             marker='x', markevery=(0.0, 0.1), markersize=markersize)
    plt.plot(mean_frob / mean_frob.max(), label='Frobenius Approximation',
             marker='o', markevery=(0.02, 0.1), markersize=markersize)
    plt.plot(mean_optim / mean_optim.max(), label='Optimization Criterion',
             marker='^', markevery=(0.04, 0.1), markersize=markersize)

    plt.xlabel("Iteration")
    plt.ylabel("Normalized Error / %")
    # plt.title(f"Argmin: {np.argmin(mean_gwd)} / {np.argmin(mean_frob)}")
    plt.legend()
    plt.ylim((-0.05, 1.05))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    create_plot_over_iterations()
