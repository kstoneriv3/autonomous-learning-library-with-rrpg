import numpy as np
import scipy.stats
import scipy.stats.qmc
from types import MethodType
from all.environments import GymEnvironment

class QMCEngine:
    def __init__(
        self,
        dim_reseed=1,
        dim_action=1,
        T_max=1000,
        qmc_type="halton",
        pca_matrix="ar1",
        scramble=True,
    ):
        # define qmc_engine
        self.dim_reseed = dim_reseed
        self.dim_action = dim_action
        self.T_max = T_max
        self.dim = T_max * dim_action + dim_reseed
        self.scramble = scramble
        self.qmc_type = qmc_type.lower()
        self.reseed()

        # define pca matrix
        self.pca_matrix = self._get_pca_matrix(T_max, pca_matrix)


    @staticmethod
    def _get_pca_matrix(n, pca_matrix):
        if pca_matrix == "ar0":
            return np.eye(n)
        elif pca_matrix == "ar1":
            C = np.tril(np.ones([n, n]))
            _, _, VT = np.linalg.svd(C)
            return VT.T
        elif pca_matrix == "ar2":
            C = np.tril(
                np.tile((np.arange(n + 1)[::-1] + 2) % (n + 1), n)[:-n].reshape([n, n])
            )
            _, _, VT = np.linalg.svd(C)
            return VT.T
        else:
            assert False

    def reseed(self):
        if self.qmc_type == "halton":
            self.engine = scipy.stats.qmc.Halton(self.dim, scramble=self.scramble)
        elif self.qmc_type == "sobol":
            self.engine = scipy.stats.qmc.Sobol(self.dim, scramble=self.scramble)
        elif self.qmc_type == "random":
            self.engine = None
        else:
            assert False

    def sample_reseed_noise(self, dist="uniform"):
        # Precompute all the samples for this episode. This needs to be called before
        # `self.sample_action_noise()`

        samples = self.engine.random(1) if self.engine else np.random.rand(self.dim)[None, :]
        samples = scipy.stats.norm.ppf(0.5 + (1 - 1e-10) * (samples - 0.5))  # Make them Gaussian

        # precompute (Gaussian) action noises
        action_noises = samples[0, self.dim_reseed :]
        action_noises = action_noises.reshape(self.T_max, self.dim_action)
        self.precomputed_action_noises = self.pca_matrix @ action_noises
        self.action_count = 0

        return self.transform_gaussian(samples[0, : self.dim_reseed], dist)

    def sample_action_noise(self, dist="uniform"):
        noise = self.precomputed_action_noises[self.action_count, :]
        noise = self.transform_gaussian(noise, dist)
        self.action_count += 1
        return noise

    @staticmethod
    def transform_gaussian(sample, dist):
        if dist.lower() == "guassian":
            return sample
        elif dist.lower() == "uniform":
            return scipy.stats.norm.cdf(sample)
        else:
            assert False


def reset_with_qmc(self, seed=None):
    #super().reset(seed)  # unnecessary and hard to bebug
    self.state = 0.1 * self.qmc_engine.sample_reseed_noise("uniform") - 0.05
    self.steps_beyond_done = None
    return np.array(self.state, dtype=np.float32)


class QMCCartPole_v1(GymEnvironment):
    def __init__(self, qmc_engine, device=None):
        env = "CartPole-v1"
        device = device or "cuda"
        super().__init__(env, device)
        # env here has following hierarchy
        # GymEnvironment
        # └─TimeLimit
        #   └─CartPoleEnv
        self._env.env.qmc_engine = qmc_engine
        self._env.env.reset = MethodType(reset_with_qmc, self._env.env)

