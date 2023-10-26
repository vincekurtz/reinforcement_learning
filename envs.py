##
#
# Some custom gym environments
#
##

from collections import deque
import gymnasium as gym
import numpy as np

class HistoryWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    A simple gym observation wrapper that outputs a history of observations
    instead of just the current one. Similar to the `FrameStack` wrapper, but 
    the buffer starts with zeros rather than copies of the initial observation.

    Note: if the original observation is of size (N,), then the wrapped
    observation will be of size, (N*history_length,).
    """
    def __init__(self, env, history_length):
        """
        Observation wrapper that holds a rolling history of past observations.

        Args:
            env: The environment to wrap.
            history_length: The length of the observation history.
        """
        # The environment must have vector observations
        assert isinstance(env.observation_space, gym.spaces.Box)    
        assert len(env.observation_space.shape) == 1

        gym.utils.RecordConstructorArgs.__init__(self, 
                                                 history_length=history_length)
        gym.ObservationWrapper.__init__(self, env)

        self.history_length = history_length
        self.frames = deque(maxlen=history_length)

        low = np.repeat(self.observation_space.low[None, ...], 
                        history_length, axis=0)
        high = np.repeat(self.observation_space.high[None, ...], history_length, 
                         axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.history_length):
            self.frames.append(np.zeros_like(obs))
        self.frames.append(obs)
        return self._get_obs(), info
    
    def _get_obs(self):
        return np.asarray(self.frames).flatten()

class EnvWithObservationHistory(gym.Env):
    """
    A simple gym environment wrapper that outputs a history of observations
    instead of just the current one.
    """
    def __init__(self, env_name, history_length, render_mode=None):
        super().__init__()
        self.history_length = history_length
        self.env = gym.make(env_name, render_mode=render_mode)

        # The environment must have vector observations
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        assert len(self.env.observation_space.shape) == 1

        assert history_length ==1, "debug"
        
        self.observation_size = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.observation_size*self.history_length,), 
            dtype=np.float32
        )
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        self.history = np.zeros((self.history_length, self.observation_size))
        obs, info = self.env.reset(**kwargs)
        self.history[0, :] = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history = np.roll(self.history, shift=1, axis=0)
        self.history[0, :] = obs
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return self.history.flatten()

if __name__=="__main__":
    # run some tests
    original = gym.make("Swimmer-v4")
    wrapped = HistoryWrapper(original, 3)

    obs, info = wrapped.reset()
    print(obs.T)

    for t in range(5):
        obs, _, _, _, _ = wrapped.step(wrapped.action_space.sample())
        print(obs.T)
