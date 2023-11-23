import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.linalg import cho_factor, cho_solve

from gwdicp.algorithms.abstract_classes import CloudRegistrationAlgorithm
from gwdicp.utils.utils import gwd, approx_gwd


class GWDICP(CloudRegistrationAlgorithm):
    """Python implementation of GWD-ICP"""

    def __init__(self, correspondence_threshold, max_iter, keypoint_dist, radius, th, shape_weight=1):
        self.correspondence_threshold = correspondence_threshold
        self.max_iter = max_iter

        self.keypoint_dist = keypoint_dist
        self.radius = radius

        self.th = th
        self.shape_weight = shape_weight

        self._last_result = None
        self._last_transform = None

        self.info_dict = None

    def register(self, source: np.ndarray, target: np.ndarray, init_tf: np.ndarray):

        def keypoints(pc: np.ndarray, keypoint_dist: float):
            """
            Function for extracting keypoints from a point cloud

            :param pc: point cloud as ndarray
            :param keypoint_dist: min. dist between keypoints
            :return: ndarray of keypoints
            """
            keypoints = []

            while (pc.shape[0] > 0):
                keypoint = pc[0]
                keypoints.append(keypoint)

                mask = np.linalg.norm(pc - keypoint, axis=1) > keypoint_dist
                pc = pc[mask]

            return np.array(keypoints)

        def gmm(pc: np.ndarray, keypoints: np.ndarray, radius: float):
            """
            Compute a GMM from a point cloud and keypoints for a given radius defining the vicinity of each keypoint.
            :param pc: Point cloud as ndarray
            :param keypoints: Keypoints in point cloud as ndarray
            :param radius: keypoint radius
            :return: corresponding mixture model as ndarray
            """
            gmm = np.zeros((0, 12))
            for keypoint in keypoints:
                mask = np.linalg.norm(pc - keypoint, axis=1) < radius
                inliers = pc[mask]
                if len(inliers) >= 4:
                    mean = np.mean(inliers, axis=0)
                    cov = np.cov(inliers.T)

                    eigvals, eigvecs = np.linalg.eig(cov)
                    eigvals = np.clip(eigvals, eigvals.max() * 0.001, eigvals.max())
                    eigvals = np.sqrt(eigvals)

                    cov_sqrt = eigvecs @ np.diag(eigvals) @ eigvecs.T

                    nd = np.zeros((12))
                    nd[:3] = mean
                    nd[3:] = cov_sqrt.flatten()
                    gmm = np.vstack((gmm, nd))

            return gmm

        def correspondences(source, target, correspondence_threshold, shape_weight):
            """
            Compute the correspondences between source and target point cloud

            :param source: source point cloud as ndarray
            :param target: target point cloud as ndarray
            :param correspondence_threshold: threshold for allowing correspondences
            :param shape_weight: weight of the shape part of the distance (comparison of covariance matrices)
            :return: matched components of source and target cloud, s.t. src[i] is matched to target[i]. unmatched
            components are not returned
            """
            _source = source.copy()
            _source[3:] = _source[3:] * shape_weight
            _target = target.copy()
            _target[3:] = _target[3:] * shape_weight

            tree = KDTree(_source)
            distances, indices = tree.query(_target)
            distances = distances ** 2
            mask = distances < correspondence_threshold ** 2
            indices = indices[mask]

            tgt = target[mask]
            src = source[indices]

            return src, tgt

        # Alignment
        def align(src, tgt, th, shape_weight):
            """
            Compute the alignment between source and target point cloud

            :param src: source point cloud as ndarray
            :param tgt: target point cloud as ndarray
            :param th: kernel weight, corresponding to sigma/3 in the paper.
            :param shape_weight: weight of the shape part of the distance (comparison of covariance matrices)
            :return: (translation, rotation): alignment results consisting of a tuple of translation and rotation,
            represented as (np.ndarray, scipy.spatial.transform.Rotation).
            """

            def hat(a):
                """
                Computes the skew-symmetric matrix of the input
                c.f., e.g., https://en.wikipedia.org/wiki/Hat_operator
                :param a: 3D input vector as ndarray of shape (3,) or (1,3)
                :return: corresponding skew symmetric 3x3 matrix as ndarray
                """
                # Checking if a is a 3-vector
                assert a.shape[0] == 3 or (a.shape[0] == 1 and a.shape[1] == 3)
                return np.array([[0, -a[2], a[1]],
                                 [a[2], 0, -a[0]],
                                 [-a[1], a[0], 0]])

            # if source and target are empty, return identity elements for translation and rotation
            #   note: this function is called after correspondence computation, so src and tgt have the same length
            #   note: if src and tgt are empty, no correspondences were determined.
            if src.shape[0] == 0:
                return np.zeros(3), np.eye(3)

            # Summed up terms for all points
            JTJ = np.zeros((6, 6))
            JTr = np.zeros((6, 1))

            residual_sums = []
            for s, t in zip(src, tgt):
                # Calculation of the Jacobian
                J_r = np.zeros((12, 6))
                J_r[:3, :3] = np.identity(3)
                J_r[:3, 3:] = -hat(s[:3])
                J_r[3:, 3:] = np.array([[0, s[5] + s[9], -s[4] - s[6]],
                                        [-s[5], s[10], s[3] - s[7]],
                                        [s[4], s[11] - s[3], -s[8]],
                                        [-s[9], s[8], s[3] - s[7]],
                                        [-s[10] - s[8], 0, s[4] + s[6]],
                                        [-s[11] + s[7], -s[6], s[5]],
                                        [s[6], s[11] - s[3], -s[10]],
                                        [-s[11] + s[7], -s[4], s[9]],
                                        [s[10] + s[8], -s[5] - s[9], 0]])

                J_r[3:, 3:] *= shape_weight
                J_r[3:, 3:] = -J_r[3:, 3:]
                # Calculation of residual
                residual = s - t
                residual = residual[:, np.newaxis]
                residual_sums.append((residual ** 2).mean())

                # Calculation of kernel weight
                _tmp = (th + np.linalg.norm(residual) ** 2)
                w = (th ** 2) / (_tmp * _tmp)

                # Summation of local results
                JTJ += (J_r.T * w) @ J_r
                JTr += (J_r.T * w) @ residual
            self.last_mean_residual = np.sum(residual_sums)

            # Cholesky factorization and solving
            c, low = cho_factor(JTJ)
            t_log = cho_solve((c, low), -JTr)
            t_log = t_log.flatten()  # Flatten t_lof from (N,1) to (N) array

            translation = t_log[:3]
            rotation = Rotation.from_rotvec(t_log[3:]).as_matrix()  # se(3) -> SE(3), i.e. from lie algebra to lie group

            return translation, rotation

        def transform_gmm(gmm, T):
            """
            Apply a transformation to all components of a GMM
            :param gmm: Gaussian mixture model to be transformed
            :param T: Transformation matrix as 4x4 ndarray
            :return: transformed mixture model
            """
            R_mat = T[:3, :3]
            R = Rotation.from_matrix(R_mat)
            t = T[:3, -1]

            gmm[:, :3] = R.apply(gmm[:, :3]) + t
            gmm[:, 3:] = (R_mat @ gmm[:, 3:].reshape((-1, 3, 3)) @ R_mat.T).reshape(-1, 9)
            return gmm

        # Transform source and target into their gmm representations
        source_keypoints = keypoints(source, self.keypoint_dist)
        source_gmm = gmm(source, source_keypoints, self.radius)

        target_keypoints = keypoints(target, self.keypoint_dist)
        target_gmm = gmm(target, target_keypoints, self.radius)

        # transforming according to initial guess
        target_gmm = transform_gmm(target_gmm, init_tf)

        # ICP
        T_icp = np.eye(4)
        gwd_over_time = []
        frob_over_time = []
        optim_over_time = []
        for i in range(self.max_iter):
            # Finding correspondences
            src, tgt = correspondences(source_gmm, target_gmm, self.correspondence_threshold, self.shape_weight)
            try:
                translation, rotation = align(src, tgt, self.th, self.shape_weight)
            except ValueError as e:
                print(align(src, tgt, self.th, self.shape_weight))
                raise e
            mean_gwd, mean_frob = self.compute_mean_dists(src, tgt)
            gwd_over_time.append(mean_gwd)
            frob_over_time.append(mean_frob)
            optim_over_time.append(self.last_mean_residual)
            T_est = np.eye(4)
            T_est[:3, :3] = rotation
            T_est[:3, -1] = translation

            # Updating acording to found transformation
            source_gmm = transform_gmm(source_gmm, T_est)

            # Updating total transformation
            T_icp[:3, :3] = T_est[:3, :3] @ T_icp[:3, :3]
            T_icp[:3, -1] = T_est[:3, :3] @ T_icp[:3, -1] + T_est[:3, -1]

        self.info_dict = {
            "n_source_features": len(source_gmm),
            "n_target_features": len(target_gmm),
            "mean_gwd": np.array(gwd_over_time),
            "mean_frob": np.array(frob_over_time),
            "mean_optim": np.array(optim_over_time)
        }

        self._last_transform = T_icp
        return self.get_last_transform()

    def get_last_transform(self) -> np.ndarray:
        return self._last_transform

    def get_lass_translation_and_rotation(self) -> tuple:
        raise NotImplementedError

    def get_information_about_last_iteration(self):
        return self.info_dict

    def compute_mean_dists(self, src, tgt):
        """
        Compute mean GWD and Frobenius approximation of ordered source and target clouds (ordered s.t. corresponding
        mixtures have corresponding indices)
        :param src: Source point cloud
        :param tgt: Target point cloud
        :return: (mean GWD, mean frobenius approx. GWD) as tuple of floats
        """
        src = np.array(src)
        tgt = np.array(tgt)
        gwds = []
        frobs = []
        for s, t in zip(src, tgt):
            s_mean = s[:3]
            s_cov = s[3:].reshape((3, 3))
            t_mean = t[:3]
            t_cov = t[3:].reshape((3, 3))
            gwds.append(gwd(s_mean, s_cov, t_mean, t_cov))
            frobs.append(approx_gwd(s_mean, s_cov, t_mean, t_cov, self.shape_weight))
        return np.mean(gwds), np.mean(frobs)

    def get_last_number_of_nds(self) -> int:
        """Return the number of mixture components for source and target cloud of the last computation"""
        return int(max(self.info_dict["n_source_features"], self.info_dict["n_target_features"]))
