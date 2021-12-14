import numpy as np
import torch
from torch.nn.functional import mse_loss
from all.core import State
from ._agent import Agent
from .a2c import A2CTestAgent


class QMCPG(Agent):
    """
    Quasi Monte Carlo Policy Gradient (QMCPG).

    Args:
        features (FeatureNetwork): Shared feature layers.
        v (VNetwork): Value head which approximates the state-value function.
        policy (StochasticPolicy): Policy head which outputs an action distribution.
        discount_factor (float): Discount factor for future rewards.
        min_batch_size (int): Updates will occurs when an episode ends after at least
            this many state-action pairs are seen. Set this to a large value in order
            to train on multiple episodes at once.
    """

    def __init__(
        self, features, v, policy, qmc_engine, discount_factor=0.99, min_batch_size=1, batch_reseeding=False,
    ):
        self.features = features
        self.v = v
        self.policy = policy
        self.qmc_engine = qmc_engine
        self.discount_factor = discount_factor
        self.min_batch_size = min_batch_size
        self.batch_reseeding = batch_reseeding
        self._current_batch_size = 0
        self._trajectories = []
        self._features = []
        self._log_pis = []
        self._rewards = []

    def act(self, state):
        if not self._features:
            return self._initial(state)
        if not state.done:
            return self._act(state, state.reward)
        return self._terminal(state, state.reward)

    def eval(self, state):
        return self.policy.eval(self.features.eval(state))

    def _initial(self, state):
        features = self.features(state)
        distribution = self.policy(features)
        action = distribution.sample()
        self._features = [features]
        self._log_pis.append(distribution.log_prob(action))
        return action

    def _act(self, state, reward):
        features = self.features(state)
        distribution = self.policy(features)
        # action = distribution.sample()
        action = self._sample_with_qmc(distribution)
        self._features.append(features)
        self._rewards.append(reward)
        self._log_pis.append(distribution.log_prob(action))
        return action

    def _terminal(self, state, reward):
        self._rewards.append(reward)
        features = State.array(self._features)
        rewards = torch.tensor(self._rewards, device=features.device)
        log_pis = torch.stack(self._log_pis)
        #print(torch.exp(log_pis).mean())
        self._trajectories.append((features, rewards, log_pis))
        self._current_batch_size += len(features)
        self._features = []
        self._rewards = []
        self._log_pis = []

        if self._current_batch_size >= self.min_batch_size:
            self._train()

        # have to return something
        return self.policy.no_grad(self.features.no_grad(state)).sample()

    def _train(self):
        # forward pass
        values = torch.cat(
            [self.v(features) for (features, _, _) in self._trajectories]
        )

        # forward passes for log_pis were stored during execution
        log_pis = torch.cat([log_pis for (_, _, log_pis) in self._trajectories])

        # compute targets
        targets = torch.cat(
            [
                self._compute_discounted_returns(rewards)
                for (_, rewards, _) in self._trajectories
            ]
        )
        advantages = targets - values.detach()
        discounts = self.discount_factor ** torch.arange(
            values.shape[0], device=values.device
        )

        # compute losses
        value_loss = mse_loss(values, targets)
        policy_loss = -(discounts * advantages * log_pis).mean()

        # backward pass
        self.v.reinforce(value_loss)
        self.policy.reinforce(policy_loss)
        self.features.reinforce()

        # cleanup
        self._trajectories = []
        self._current_batch_size = 0
        if self.batch_reseeding:
            self.qmc_engine.reseed()

    def _compute_discounted_returns(self, rewards):
        returns = rewards.clone()
        t = len(returns) - 1
        discounted_return = 0
        for reward in torch.flip(rewards, dims=(0,)):
            discounted_return = reward + self.discount_factor * discounted_return
            returns[t] = discounted_return
            t -= 1
        return returns

    def _sample_with_qmc(self, distribution):
        if isinstance(distribution, torch.distributions.Categorical):
            cmf = torch.cumsum(distribution.probs, dim=0)[:-1]
            noise = self.qmc_engine.sample_action_noise("uniform")
            #noise = np.random.rand(1)
            noise = torch.as_tensor(noise, device=cmf.device, dtype=cmf.dtype)
            return torch.sum(noise > cmf)
        elif isinstance(distribution, torch.distributions.Normal):
            noise = self.qmc_engine.sample_action_noise("Gaussian")
            return distribution.mean + distribution.scale * noise
        else:
            assert False


QMCPGTestAgent = A2CTestAgent
