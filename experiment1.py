import gym
import d4rl

env = gym.make('antmaze-medium-play-v0')
print(env.observation_space.shape, env.action_space.shape)