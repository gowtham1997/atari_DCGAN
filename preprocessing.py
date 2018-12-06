import gym
import gym.spaces
import numpy as np
import cv2
from torch import tensor
import random


class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args, img_size=64):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.IMG_SIZE = img_size
        # redefine new subspace
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32)
        # print(old_space, self.observation_space)

    # override observation(obs) method
    def observation(self, obs):
        new_obs = cv2.resize(obs, (self.IMG_SIZE, self.IMG_SIZE))
        # (h, w, d) -> (d, w, h)
        new_obs = np.moveaxis(new_obs, 2, 0)
        # restrict image pixels to (0, 1)
        return new_obs.astype(np.float32)


def iterate_batches(envs, batch_size):
    # infinte env sampler.
    # random.choice will never be None, hence this will sample infinitely
    env_gen = iter(lambda: random.choice(envs), None)
    batch = [e.reset() for e in envs]
    while True:
        # env = env_gen.__next__()
        env = next(env_gen)
        obs, action, done, _ = env.step(env.action_space.sample())
        # sometimes the pic is 0 completely due to flikcering
        if np.mean(obs) < 0.01:
            continue
        batch.append(obs)
        if len(batch) == batch_size:
            batch_np = (np.array(batch, dtype=np.float32) - 127.5) / 127.5
            yield tensor(batch_np)
            batch.clear()
        if done:
            env.reset()
