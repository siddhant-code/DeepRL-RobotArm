from gymnasium.envs.registration import register

register(
    id='robot-arm-v1',
    entry_point='baxter_robot.envs:BaxterEnv',
)
