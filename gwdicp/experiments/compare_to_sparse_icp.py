"""
Main simulation analysis. Comparison of GWD-ICP to point-to-point ICP, with different levels of sparsity used for the
standard ICP approach.
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from tqdm import tqdm

from data_generation import draw_multilayer_samples_from_env, add_noise
from gwdicp.algorithms.open3d_icp_wrapper import FullOpen3DICP
from gwdicp.algorithms.abstract_classes import SubsamplingWrapper
from gwdicp.algorithms.gwd_icp_implementation import GWDICP
from gwdicp.experiments.configs import CONFIG_DEFAULT
from gwdicp.utils.utils import transform_cloud, compute_error_statistics, compute_mean_point_error


def main():
    plt.style.use("../../data/stylesheets/paper.mplstyle")

    seed = 42  # seed for RNG
    disable_tqdm = False  # set to True to disable tqdm
    add_std_to_plot = True  # whether to shade the std dev in the plot

    config = CONFIG_DEFAULT

    # config["n_monte_carlo_runs_per_setting"] = 100

    icp_instance = FullOpen3DICP(mode="point-point", correspondence_threshold=10, max_iter=1000)

    icp_point_count_factors = [1,  # same number of points as NDs
                               3,  # same space requirements as NDs (mean = 1 point, cov=sym. 3x3 = 6D = 2pts)
                               *[3 * x for x in range(1, 11)]]  # upscale space req. linearly

    gwd_icp = GWDICP(correspondence_threshold=10, max_iter=100, keypoint_dist=2.5, radius=1.5, th=0.75)

    print(f"Loaded config")
    print("")

    # load RNG
    if seed is None:
        seed = np.random.randint(low=1, high=99999)
        print(f"Setting seed to {seed}")
    rng = np.random.default_rng(seed)

    errors, nd_counts = compute_method_errors(rng=rng,
                                              gwd_icp_algorithm=gwd_icp,
                                              icp_base_algorithm=icp_instance,
                                              icp_point_count_factors=icp_point_count_factors,
                                              offset_centers=config["offset_centers"],
                                              sensor_radius=config["sensor_range"],
                                              gt_transform=config["gt_transform"],
                                              center_point=config["center_point"],
                                              env_lines=config["environment"],
                                              sample_density=config["sample_density"],
                                              n_layers=config["n_layers"],
                                              max_height=config["max_height"],
                                              meas_cov=config["meas_cov"],
                                              n_monte_carlo_runs=config["n_monte_carlo_runs_per_setting"],
                                              disable_tqdm=disable_tqdm)

    print("Normal Distribution Counts:")
    print(f"\tMean  : {nd_counts.mean()}")
    print(f"\tMin   : {nd_counts.min()}")
    print(f"\tMax   : {nd_counts.max()}")
    print(f"\tMedian: {np.median(nd_counts)}")

    # plot_single_error(errors, icp_point_count_factors, add_std_to_plot, True)
    # plot_three_errors(errors, icp_point_count_factors, add_std_to_plot, True)
    plot_three_errors_separate(errors, icp_point_count_factors, add_std_to_plot, False)


def compute_method_errors(rng: np.random.Generator,
                          gwd_icp_algorithm,
                          icp_base_algorithm,
                          icp_point_count_factors,
                          offset_centers: np.ndarray,
                          sensor_radius: float,
                          gt_transform: np.ndarray,
                          center_point: np.ndarray,
                          env_lines: np.ndarray,
                          sample_density: float,
                          n_layers: int,
                          max_height: float,
                          meas_cov: np.ndarray,
                          n_monte_carlo_runs: int,
                          disable_tqdm: bool,
                          ):
    """
    Compute errors (translation, rotation) for a set of icp variants given parameters. The errors are computed
    for a number of monte carlo runs using the given settings. Uses one GWD-ICP instance and creates subsampling-based
    ICP variants which use a number of points corresponding to the number of NDs used by GWD-ICP

    :param rng: Generator to use
    :param gwd_icp_algorithm: GWD ICP instance
    :param icp_base_algorithm: Base icp copied for all subsampling wrappers of ICP
    :param icp_point_count_factors: Values used for subsampling the point cloud before point to point ICP. Multiplied
    with the number of ellipses used by gwd_icp
    :param offset_centers: 3D Offset between the centers of the point cloud, i.e., the second point clouds is centered
    on offset_centers + center_point
    :param sensor_radius: Each point cloud will only consist of points that are at most this far from the point cloud
    center
    :param gt_transform: 4x4 Transform that should be matched, i.e., translation and rotation matching the two point
    clouds
    :param center_point: 3D center point of the first point cloud
    :param env_lines: Point-based line definition used for point sampling
    :param sample_density: Points per meter on lines in environment
    :param n_layers: Number of vertical layers to project points to
    :param max_height: Maximum height to which points are distributed
    :param meas_cov: 3x3 measurement noise covariance matrix added to sampled points
    :param n_monte_carlo_runs: Number of monte carlo runs to perform
    :param disable_tqdm: Whether to disable tqdm or not
    :return: dict[str, np.ndarray]: Mapping algorithm names onto an array of shape (n_monte_carlo_runs, 2) errors
    consisting of translation and rotation errors each
    """
    # prepare function to be used for point generation based on parameter values
    vertical_distance = max_height / n_layers

    def pt_generation_fct():
        original_samples = draw_multilayer_samples_from_env(
            rng=rng,
            line_segment_list=np.array(env_lines),
            sample_density=sample_density,
            n_layers=n_layers,
            dist_between_layers=vertical_distance
        )
        return add_noise(rng, original_samples, meas_cov)

    # prepare error dict
    errors = {a: [] for a in ["GWD-ICP", *[f"ICP-{x}" for x in icp_point_count_factors]]}
    nd_counts = []

    # perform monte carlo runs
    for mc_ix in tqdm(range(n_monte_carlo_runs), disable=disable_tqdm, total=n_monte_carlo_runs):
        noisy_samples = pt_generation_fct()
        init_guess = np.zeros((4, 4))
        init_guess[3, 3] = 1
        init_guess[:3, :3] = np.eye(3)
        center_point_shifted = offset_centers + gt_transform[:3, 3]
        shifted_pts = transform_cloud(pt_generation_fct() - center_point, gt_transform)
        noisy_samples = noisy_samples - center_point
        noisy_samples = noisy_samples[np.linalg.norm(noisy_samples - center_point, axis=1) < sensor_radius]
        shifted_pts = shifted_pts[np.linalg.norm(shifted_pts - center_point_shifted, axis=1) < sensor_radius]

        # save errors
        # start with gwd-icp
        est_tf = gwd_icp_algorithm.register(noisy_samples, shifted_pts, init_guess)
        next_error = [
            *compute_error_statistics(est_tf, gt_transform),
            compute_mean_point_error(est_tf, gt_transform)
        ]
        errors["GWD-ICP"].append(next_error)

        normal_distribution_count = gwd_icp_algorithm.get_last_number_of_nds()
        nd_counts.append(normal_distribution_count)

        for point_factor in icp_point_count_factors:
            algorithm = SubsamplingWrapper(rng=rng, subsampling_value=point_factor * normal_distribution_count,
                                           wrapped_method=icp_base_algorithm)
            est_tf = algorithm.register(noisy_samples, shifted_pts, init_guess)
            next_error = [
                *compute_error_statistics(est_tf, gt_transform),
                compute_mean_point_error(est_tf, gt_transform)
            ]
            errors[f"ICP-{point_factor}"].append(next_error)

    # convert to numpy and return
    errors = {a: np.array(errors[a]) for a in errors.keys()}
    return errors, np.asarray(nd_counts)


def plot_single_error(error_dict: dict,
                      icp_point_count_factors,
                      show_std,
                      add_vertical_line_at_icp3):
    error_dict = deepcopy(error_dict)
    ix_error = 2
    color_gwd = 'blue'
    color_icp = 'orange'
    std_alpha = .2

    error_gwd_icp = error_dict.pop("GWD-ICP")
    errors_for_point_count_factors = np.asarray([
        error_dict[f"ICP-{p}"][:, ix_error].mean()
        for p in icp_point_count_factors
    ])
    plt.plot(icp_point_count_factors,
             errors_for_point_count_factors,
             marker='x', linestyle='--', label='ICP',
             c=color_icp)
    plt.axhline(error_gwd_icp[:, ix_error].mean(), label='GWD-ICP')

    if show_std:
        gwd_icp_std = np.std(error_gwd_icp[:, ix_error])
        plt.axhspan(error_gwd_icp[:, ix_error].mean() - gwd_icp_std,
                    error_gwd_icp[:, ix_error].mean() + gwd_icp_std,
                    facecolor=color_gwd,
                    alpha=std_alpha)

        icp_std_array = np.asarray([
            error_dict[f"ICP-{p}"][:, ix_error].std() for p in icp_point_count_factors
        ])
        plt.fill_between(icp_point_count_factors,
                         errors_for_point_count_factors - icp_std_array,
                         errors_for_point_count_factors + icp_std_array,
                         color=color_icp,
                         alpha=std_alpha)

    xlim = plt.xlim()
    ylim = plt.ylim()

    if add_vertical_line_at_icp3:
        plt.vlines(3, -1, error_dict[f"ICP-3"][:, ix_error].mean(), color='red')

    plt.xlabel("Number of ICP Points / Number of GWD-ICP Normal Distributions")
    plt.ylabel("Average Error / m")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.show()


def plot_three_errors(error_dict: dict,
                      icp_point_count_factors,
                      show_std,
                      add_vertical_line_at_icp3):
    error_dict = deepcopy(error_dict)
    color_gwd = 'blue'
    color_icp = 'orange'
    std_alpha = .2

    fig, axs = plt.subplots(1, 3, figsize=(17, 6))

    y_labels = [
        "Translation error / m",
        "Rotation error / rad",
        "Average Error / m"
    ]

    titles = [
        "Translation Error",
        "Rotation Error",
        "Average Displacement Error"
    ]
    error_gwd_icp = error_dict.pop("GWD-ICP")
    for ix_error in range(3):
        plt.sca(axs[ix_error])

        errors_for_point_count_factors = np.asarray([
            error_dict[f"ICP-{p}"][:, ix_error].mean()
            for p in icp_point_count_factors
        ])
        plt.plot(icp_point_count_factors,
                 errors_for_point_count_factors,
                 marker='x', linestyle='--', label='ICP',
                 c=color_icp)
        plt.axhline(error_gwd_icp[:, ix_error].mean(), label='GWD-ICP')

        if show_std:
            gwd_icp_std = np.std(error_gwd_icp[:, ix_error])
            plt.axhspan(error_gwd_icp[:, ix_error].mean() - gwd_icp_std,
                        error_gwd_icp[:, ix_error].mean() + gwd_icp_std,
                        facecolor=color_gwd,
                        alpha=std_alpha)

            icp_std_array = np.asarray([
                error_dict[f"ICP-{p}"][:, ix_error].std() for p in icp_point_count_factors
            ])
            plt.fill_between(icp_point_count_factors,
                             errors_for_point_count_factors - icp_std_array,
                             errors_for_point_count_factors + icp_std_array,
                             color=color_icp,
                             alpha=std_alpha)

        xlim = plt.xlim()
        ylim = plt.ylim()

        if add_vertical_line_at_icp3:
            plt.vlines(3, -1, error_dict[f"ICP-3"][:, ix_error].mean(), color='red')

        plt.xlabel("Number of ICP Points / Number of GWD-ICP Normal Distributions")
        plt.xticks(icp_point_count_factors, icp_point_count_factors)
        plt.ylabel(y_labels[ix_error])
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(titles[ix_error])
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_three_errors_separate(error_dict: dict,
                               icp_point_count_factors,
                               show_std,
                               add_vertical_line_at_icp3):
    error_dict = deepcopy(error_dict)
    color_gwd = 'blue'
    color_icp = 'orange'
    std_alpha = .2

    y_labels = [
        "Translation error / m",
        "Rotation error / rad",
        "Average displacement error / m"
    ]

    titles = [
        "Translation Error",
        "Rotation Error",
        "Average Displacement Error"
    ]
    error_gwd_icp = error_dict.pop("GWD-ICP")
    for ix_error in range(3):
        errors_for_point_count_factors = np.asarray([
            error_dict[f"ICP-{p}"][:, ix_error].mean()
            for p in icp_point_count_factors
        ])
        plt.plot(icp_point_count_factors,
                 errors_for_point_count_factors,
                 marker='x',
                 label='ICP', c=color_icp)
        plt.axhline(error_gwd_icp[:, ix_error].mean(), label='GWD-ICP')

        if show_std:
            gwd_icp_std = np.std(error_gwd_icp[:, ix_error])
            plt.axhspan(error_gwd_icp[:, ix_error].mean() - gwd_icp_std,
                        error_gwd_icp[:, ix_error].mean() + gwd_icp_std,
                        facecolor=color_gwd,
                        alpha=std_alpha)

            icp_std_array = np.asarray([
                error_dict[f"ICP-{p}"][:, ix_error].std() for p in icp_point_count_factors
            ])
            plt.fill_between(icp_point_count_factors,
                             errors_for_point_count_factors - icp_std_array,
                             errors_for_point_count_factors + icp_std_array,
                             color=color_icp,
                             alpha=std_alpha)

        xlim = plt.xlim()
        ylim = plt.ylim()

        if add_vertical_line_at_icp3:
            plt.vlines(3, -1, error_dict[f"ICP-3"][:, ix_error].mean(), color='red')

        # plt.xlabel("ICP Point Count Scaling")
        plt.xlabel(r"$\lambda$")
        plt.xticks(icp_point_count_factors, icp_point_count_factors)
        plt.ylabel(y_labels[ix_error])
        plt.xlim(xlim)
        plt.ylim(ylim)
        # plt.title(titles[ix_error])
        plt.legend()
        plt.tight_layout()
        name = "icp_scaling_" + titles[ix_error].lower().replace(" ", "_")
        plt.savefig(f"../../output/paper/{name}.svg")
        plt.close()


if __name__ == '__main__':
    main()
