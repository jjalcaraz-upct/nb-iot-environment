from gymnasium.envs.registration import register

from .environment import Environment

register(
    id='System-v1',
    entry_point='gym_system:Environment'
)