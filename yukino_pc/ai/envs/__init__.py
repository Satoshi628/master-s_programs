from gym.envs.registration import register

register(
    id='pysim2d-v0',
    entry_point='myenv:MyEnv',
    max_episode_steps=10000,
)
