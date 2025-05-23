# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class SimpleForwardTask(object):
    """Default empy task."""

    def __init__(self, target_vel: float):
        """Initializes the task."""
        self.target_vel = target_vel
        self.sigma = 0.33
        self.current_vel = 0
        self.current_base_pos = np.zeros(3)
        self.last_base_pos = np.zeros(3)

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env
        self.last_base_pos = env.robot.GetBasePosition()
        self.current_base_pos = self.last_base_pos
        self.current_vel = 0

    def update(self, env):
        """Updates the internal state of the task."""
        self.last_base_pos = self.current_base_pos
        self.current_base_pos = env.robot.GetBasePosition()
        self.current_vel = self.current_base_pos[0] - self.last_base_pos[0]

    def done(self, env):
        """Checks if the episode is over.

        If the robot base becomes unstable (based on orientation), the episode
        terminates early.
        """
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        return rot_mat[-1] < 0.85

    def reward(self, env):
        """Get the reward without side effects."""
        del env
        return np.exp(-((self.current_vel - self.target_vel) ** 2) / self.sigma)
