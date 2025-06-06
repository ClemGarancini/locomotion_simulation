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
"""Utilities for building environments."""
from locomotion.envs import locomotion_gym_env
from locomotion.envs import locomotion_gym_config
from locomotion.envs.env_wrappers import (
    observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper,
)
from locomotion.envs.env_wrappers import trajectory_generator_wrapper_env
from locomotion.envs.env_wrappers import simple_openloop
from locomotion.envs.env_wrappers import simple_forward_task
from locomotion.envs.sensors import robot_sensors
from locomotion.robots import a1
from locomotion.robots import laikago
from locomotion.robots import robot_config


def build_regular_env(
    robot_class,
    motor_control_mode,
    enable_rendering=False,
    on_rack=False,
    action_limit=(0.75, 0.75, 0.75),
    wrap_trajectory_generator=True,
):

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    sim_params.motor_control_mode = motor_control_mode
    sim_params.reset_time = 2
    sim_params.num_action_repeat = 10
    sim_params.enable_action_interpolation = False
    sim_params.enable_action_filter = False
    sim_params.enable_clip_motor_commands = False
    sim_params.robot_on_rack = on_rack

    gym_config = locomotion_gym_config.LocomotionGymConfig(
        simulation_parameters=sim_params
    )

    sensors = [
        robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS),
        robot_sensors.MotorVelocitySensor(num_motors=a1.NUM_MOTORS),
        robot_sensors.BaseDisplacementSensor(channels=["x"]),
        robot_sensors.IMUSensor(channels=["R", "P", "dR", "dP", "dY"]),
        robot_sensors.FootContactSensor(),
    ]

    task = simple_forward_task.SimpleForwardTask(target_vel=1.0)

    env = locomotion_gym_env.LocomotionGymEnv(
        gym_config=gym_config, robot_class=robot_class, robot_sensors=sensors, task=task
    )

    env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    if (
        motor_control_mode == robot_config.MotorControlMode.POSITION
    ) and wrap_trajectory_generator:
        if robot_class == laikago.Laikago:
            env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
                env,
                trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
                    action_limit=action_limit
                ),
            )
        elif robot_class == a1.A1:
            env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
                env,
                trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
                    action_limit=action_limit
                ),
            )
    return env
