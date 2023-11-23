import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation


def tuple_to_transformation(translation: np.ndarray, euler_angle_rotation: np.ndarray):
    """
    Given translation and rotation, return 4x4 transformation matrix
    :param translation: x y z
    :param euler_angle_rotation: pitch yaw roll in radians
    :return: 4x4 Transformation
    """
    tf = np.zeros((4, 4))
    tf[:3, :3] = Rotation.from_euler("yzx", angles=euler_angle_rotation).as_matrix()
    tf[:3, 3] = translation
    tf[3, 3] = 1
    return tf


def transform_cloud(cloud: np.ndarray,
                    transform: np.ndarray,
                    rotation_center: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
    """
    Given a numpy point cloud and a 4x4 transform, apply the rotation and translation to the cloud
    :param cloud: Nx3 array of xyz coordinates
    :param transform: 4x4 tf matrix
    :param rotation_center: Point around which rotation is performed
    :return: Transformed cloud as Nx3 array
    """
    # ensure numpy and copy to prevent working in-place
    cloud = np.array(cloud)
    # rotate every row by rotation matrix using einsum
    cloud -= rotation_center
    cloud = np.einsum('jk,ik->ij', transform[:3, :3], cloud)
    cloud += rotation_center
    # translate
    cloud += transform[:3, 3]
    return cloud


def compute_error_statistics(est_tf: np.ndarray, true_tf: np.ndarray) -> np.ndarray:
    """
    Given two 4x4 transformation matrices, return the translation error in meters and the rotation error in radians
    as a (2,) ndarray
    :param est_tf: 4x4 ndarray of estimated transform
    :param true_tf: 4x4 ndarray of true transform
    :return: [translation, rotation] error
    """
    # return compute_error_statistics_from_point_transformations(est_tf, true_tf)  # USE ALT. ERROR
    # dist = norm of diff vector between translations
    dist = np.linalg.norm(est_tf[:3, 3] - true_tf[:3, 3])

    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    theta = np.arccos((np.trace(est_tf[:3, :3] @ true_tf[:3, :3].T) - 1) / 2)
    return np.asarray([dist, theta])


def compute_mean_point_error(est_tf: np.ndarray, true_tf: np.ndarray) -> np.ndarray:
    """Shift points according to one tf, return using other tf, compare distances"""
    pts = np.array([
        [-20, 2], [-7, 2],
        [-7, 2], [-7, 4],
        [-4, 4], [-4, 0],
        [-20, -2], [-14, -2],
        [-14, -2], [-14, -6],
        [-11, -2], [-11, -6],
        [-11, -2], [-4, -2],
    ])

    pts = np.hstack([pts, np.ones((len(pts), 1))])

    test_pts = transform_cloud(np.array(pts), true_tf)
    test_pts = transform_cloud(test_pts, invert_tf(est_tf))
    return np.linalg.norm(test_pts - pts, axis=1).mean()


def invert_tf(tf):
    if tf.shape == (4, 4):
        new_tf = np.array(tf)
        new_tf[:3, 3] = -tf[:3, 3]
        new_tf[:3, :3] = tf[:3, :3].T
    elif tf.shape == (3, 3):
        new_tf = np.array(tf)
        new_tf[:2, 2] = -tf[:2, 2]
        new_tf[:2, :2] = tf[:2, :2].T
    else:
        raise ValueError(f"Invalid transform shape {tf.shape}")
    return new_tf


def approx_gwd(m1, C1, m2, C2, shape_weight=1):
    """Compute the approximate GWD using Frobenius norm"""
    center_part = np.linalg.norm(m1 - m2) ** 2
    shape_part = np.linalg.norm((sqrtm(C1) - sqrtm(C2) * shape_weight), ord='fro') ** 2
    d = center_part + shape_part
    return d


def gwd(m1, C1, m2, C2):
    center_part = np.linalg.norm(m1 - m2) ** 2
    shape_part = np.trace(C1 + C2 - 2 * sqrtm(sqrtm(C1) @ C2 @ sqrtm(C1)))
    d = center_part + shape_part
    return d
