from gym.envs.registration import register
register(
    id='ballcatch-v0',
    entry_point='gym_thing.envs:ThingBallCatchEnv',
    max_episode_steps=1000,
)

register(
    id='thing-v0',
    entry_point='gym_thing.envs:ThingEnv',
)

