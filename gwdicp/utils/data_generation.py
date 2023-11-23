import numpy as np


def draw_samples_from_env(rng: np.random.Generator,
                          line_segment_list: np.ndarray,
                          sample_density: float):
    """
    Given a set of line segments, randomly samples points from them and returns the points
    :param rng: Generator object used for sampling
    :param line_segment_list: Nx4 array of lines, each entry is [x1, y2, x2, y2]
    :param sample_density: Density in points per meter to sample.
    :return: Px2 array of P sampled points on the lines
    """
    locations = []
    for line in line_segment_list:
        p1, p2 = line[:2], line[2:]
        n_pts_on_line = sample_density * np.linalg.norm(p2 - p1)
        scale = rng.random(int(n_pts_on_line))
        locations.append(p1 + (scale * (p2 - p1)[:, np.newaxis]).T)
    locations = np.vstack(locations)
    rng.shuffle(locations)
    return locations


def add_noise(rng: np.random.Generator,
              pts: np.ndarray,
              cov: np.ndarray):
    """
    Given a set of 2D points, add Gaussian noise to each point and return the resulting estimate
    :param rng: Generator object used for sampling
    :param pts: Nxd array of points used as "true" locations
    :param cov: dxd covariance matrix of the zero-mean Gaussian noise added to pts
    :return: Noisy pts as Nxd array
    """
    assert cov.shape == (pts.shape[1], pts.shape[1]), f"Cov. shape is {cov.shape} but points are {pts.shape[1]}-d!"
    return rng.multivariate_normal(np.zeros((pts.shape[1],)), cov, size=pts.shape[0]) + np.array(pts)


def draw_multilayer_samples_from_env(rng: np.random.Generator,
                                     line_segment_list: np.ndarray,
                                     sample_density: float,
                                     n_layers: int,
                                     dist_between_layers: float):
    """
    Given a set of line segments, randomly samples points from them and returns the points. Samples from these points
    from a set of layers on the z-Axis, starting at z=0.
    :param rng: Generator object used for sampling
    :param line_segment_list: Nx4 array of lines, each entry is [x1, y2, x2, y2]
    :param sample_density: Density in points per meter to sample
    :param n_layers: Number of layers to use
    :param dist_between_layers: Vertical distance between layers
    :return: Px3 array of P sampled points on the lines
    """
    all_pts = np.zeros((0, 3))
    for layer_ix in range(n_layers):
        next_samples = draw_samples_from_env(rng, line_segment_list, sample_density)
        all_pts = np.vstack([
            all_pts,
            np.hstack([next_samples, np.full((len(next_samples), 1), fill_value=dist_between_layers * layer_ix)])
        ])
    return all_pts
