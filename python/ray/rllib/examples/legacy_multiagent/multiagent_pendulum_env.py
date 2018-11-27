from gym.spaces import Tuple, Box
from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np
"""
 Multiagent pendulum that has two identical pendulums being operated
 by separate agents. Their rewards are summed as we currently only
 support shared rewards. Actions are part of the state space so that
 they get passed to the value function
"""


class DoubleMultiAgentPendulumEnv(PendulumEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = [
            Box(low=-self.max_torque / 2,
                high=self.max_torque / 2,
                shape=(1, ),
                dtype=np.float32) for _ in range(2)
        ]
        self.observation_space = Tuple(
            [Box(low=-high, high=high, dtype=np.float32) for _ in range(2)])

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        state = []
        costs = 0
        actions = []
        for i in range(2):
            th, thdot = self.state[i]  # th := theta

            action = u[i][0]
            actions.append(action)
            g = 10.
            m = 1.
            length = 1.
            dt = self.dt

            action = np.clip(action, -self.max_torque, self.max_torque)
            self.last_u = action  # for rendering
            costs += self.angle_normalize(th) ** 2 + .1 * thdot ** 2 + \
                .001 * (action ** 2)

            newthdot = thdot + (-3 * g / (2 * length) * np.sin(th + np.pi) +
                                3. / (m * length ** 2) * action) * dt
            newth = th + newthdot * dt
            newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        newthdot = thdot + (-3 * g / (2 * length) * np.sin(th + np.pi) + 3. /
                            (m * length**2) * summed_u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = state
        return self._get_obs(actions), -costs[0]/2, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = [self.np_random.uniform(low=-high, high=high)
                      for _ in range(2)]
        self.last_u = None
        return self._get_obs([0.0, 0.0])

    def _get_obs(self):
        theta, thetadot = self.state
        return [
            np.array([np.cos(theta), np.sin(theta), thetadot])
            for _ in range(2)
        ]

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi
