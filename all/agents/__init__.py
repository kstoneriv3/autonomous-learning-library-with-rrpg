from ._agent import Agent
from ._multiagent import Multiagent
from ._parallel_agent import ParallelAgent
from .a2c import A2C, A2CTestAgent
from .c51 import C51, C51TestAgent
from .ddpg import DDPG, DDPGTestAgent
from .ddqn import DDQN, DDQNTestAgent
from .dqn import DQN, DQNTestAgent
from .independent import IndependentMultiagent
from .ppo import PPO, PPOTestAgent
from .qmcpg import QMCPG, QMCPGTestAgent
from .rainbow import Rainbow, RainbowTestAgent
from .rrpg import RRPG, RRPGTestAgent
from .sac import SAC, SACTestAgent
from .vac import VAC, VACTestAgent
from .vpg import VPG, VPGTestAgent
from .vqn import VQN, VQNTestAgent
from .vsarsa import VSarsa, VSarsaTestAgent


__all__ = [
    # Agent interfaces
    "Agent",
    "Multiagent",
    "ParallelAgent",
    # Agent implementations
    "A2C",
    "A2CTestAgent",
    "C51",
    "C51TestAgent",
    "DDPG",
    "DDPGTestAgent",
    "DDQN",
    "DDQNTestAgent",
    "DQN",
    "DQNTestAgent",
    "PPO",
    "PPOTestAgent",
    "QMCPG",
    "QMCPGTestAgent",
    "Rainbow",
    "RainbowTestAgent",
    "RRPG",
    "RRPGTestAgent",
    "SAC",
    "SACTestAgent",
    "VAC",
    "VACTestAgent",
    "VPG",
    "VPGTestAgent",
    "VQN",
    "VQNTestAgent",
    "VSarsa",
    "VSarsaTestAgent",
    "IndependentMultiagent",
]
