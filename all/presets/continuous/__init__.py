# from .actor_critic import actor_critic
from .ddpg import ddpg, DDPGContinuousPreset
from .ppo import ppo, PPOContinuousPreset
from .sac import sac
from .vpg import vpg, VPGContinuousPreset

__all__ = [
    'ddpg',
    'ppo',
    'sac',
    'vpg',
]
