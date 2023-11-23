import numpy as np
import open3d as o3d
from algorithms.abstract_classes import CloudRegistrationAlgorithm


class FullOpen3DICP(CloudRegistrationAlgorithm):
    """Wraps around the o3d ICP pipeline"""
    def __init__(self, correspondence_threshold, max_iter,
                 mode='point-point'):
        self.mode = mode

        self.correspondence_threshold = correspondence_threshold
        self.max_iter = max_iter

        if self.mode == "point-point":
            self.icp_mode = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        elif self.mode == "point-plane":
            self.icp_mode = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        elif self.mode == "generalized":
            self.icp_mode = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        else:
            raise ValueError(f"Unknown registration mode '{self.mode}'")

        self.convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iter)

        self.search_params = {
            "radius": 0.1,
            "max_nn": 30
        }

        self._last_result = None
        self._last_transform = None

    def register(self, source: np.ndarray, target: np.ndarray, init_tf: np.ndarray):
        source_o3d_cloud = o3d.geometry.PointCloud()
        source_o3d_cloud.points = o3d.utility.Vector3dVector(source)

        target_o3d_cloud = o3d.geometry.PointCloud()
        target_o3d_cloud.points = o3d.utility.Vector3dVector(target)

        # optional normal estimation if necessary
        if self.mode == "point-plane" or self.mode == "generalized":
            source_o3d_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(**self.search_params))
            target_o3d_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(**self.search_params))

        self._last_result = o3d.pipelines.registration.registration_icp(source_o3d_cloud, target_o3d_cloud,
                                                                        self.correspondence_threshold, init_tf,
                                                                        self.icp_mode,
                                                                        self.convergence_criteria)
        self._last_transform = self._last_result.transformation

        return self.get_last_transform()

    def get_last_transform(self) -> np.ndarray:
        return self._last_transform

    def get_lass_translation_and_rotation(self) -> tuple:
        raise NotImplementedError
