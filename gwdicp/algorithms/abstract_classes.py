import numpy as np

from abc import ABC, abstractmethod


class CloudRegistrationAlgorithm(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def register(self, source: np.ndarray, target: np.ndarray, init_tf) -> np.ndarray:
        """
        Register two point clouds with each other
        :param source: Source point cloud as Nx3 numpy array
        :param target: Target point cloud as Mx3 numpy array
        :param init_tf: Initial transform as 4x4 numpy array
        :return: Transform result as 4x4 numpy array
        """
        pass
        return self.get_last_transform()

    @abstractmethod
    def get_last_transform(self) -> np.ndarray:
        """
        Return the last transform result as 4x4 numpy array
        :return: Last transform result as 4x4 numpy array
        """
        pass

    @abstractmethod
    def get_lass_translation_and_rotation(self) -> tuple:
        pass


class SubsamplingWrapper(CloudRegistrationAlgorithm):
    def __init__(self,
                 subsampling_value,
                 rng: np.random.Generator,
                 wrapped_method: CloudRegistrationAlgorithm):
        """
        Wraps around an existing algorithm, subsampling points before passing on the point clouds
        :param subsampling_value: Either float in (0, 1] (percent based subsampling) or int indicating how many pts to
        sample to
        :param rng: RNG object used for sampling
        :param wrapped_method: Any CloudRegistrationAlgorithm instance
        """
        assert subsampling_value > 0
        self.subsampling_value = subsampling_value
        self.rng = rng
        self.wrapped_method = wrapped_method

    def register(self, source: np.ndarray, target: np.ndarray, init_tf) -> np.ndarray:
        """
        Register two point clouds with each other
        :param source: Source point cloud as Nx3 numpy array
        :param target: Target point cloud as Mx3 numpy array
        :param init_tf: Initial transform as 4x4 numpy array
        :return: Transform result as 4x4 numpy array
        """
        # Determine n_pts to subsample to
        if self.subsampling_value <= 1:
            source_n_pts = int(len(source) * self.subsampling_value)
            target_n_pts = int(len(target) * self.subsampling_value)
        else:
            source_n_pts = int(min(self.subsampling_value, len(source)))
            target_n_pts = int(min(self.subsampling_value, len(target)))

        # subsample
        source_ix_to_keep = self.rng.choice(len(source),
                                            size=source_n_pts,
                                            replace=True)
        source = np.array(source[source_ix_to_keep, :])

        target_ix_to_keep = self.rng.choice(len(target),
                                            size=target_n_pts,
                                            replace=True)
        target = np.array(target[target_ix_to_keep, :])
        return self.wrapped_method.register(source, target, init_tf)

    def get_last_transform(self) -> np.ndarray:
        """
        Return the last transform result as 4x4 numpy array
        :return: Last transform result as 4x4 numpy array
        """
        return self.wrapped_method.get_last_transform()

    def get_lass_translation_and_rotation(self) -> tuple:
        return self.wrapped_method.get_lass_translation_and_rotation()
