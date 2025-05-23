from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import argparse
from datetime import datetime
import os
import json
import shutil
import gym
from pathlib import Path

def load_config(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def main(config_path: str):
    config = load_config(config_path)
    # num_envs = config["environment"]["num_envs"]
    total_timesteps = config["training"]["total_timesteps"]
    checkpoint_freq = config["training"]["checkpoint_timesteps"]
    checkpoint_freq = checkpoint_freq

    task = "locomotion:A1GymEnv-v1"
    path_to_logs = Path(__file__).resolve().parent / "logs"


    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logfile = "PPO_" + task + "_" + str(timestamp)
    logfile_monitor = os.path.join(
        f"{path_to_logs}/monitor",
        logfile,
        "",
    )
    logfile_model = os.path.join(
        f"{path_to_logs}/models",
        logfile,
    )
    logfile_parameters = os.path.join(
        f"{path_to_logs}/parameters",
        logfile + ".json",
    )
    logfile_checkpoints = (
        f"{path_to_logs}/checkpoints"
    )
    os.makedirs(logfile_monitor, exist_ok=True)
    os.makedirs(logfile_model, exist_ok=True)
    os.makedirs(logfile_checkpoints, exist_ok=True)
    os.makedirs(logfile_parameters, exist_ok=True)
    print("Making gym envs...")

    env = gym.make(task)
    env = Monitor(env, filename=logfile_monitor)

    print("Done")

    model = PPO("MlpPolicy", env=env, verbose=1, n_steps=512)

    shutil.copy(config_path, logfile_parameters)
    print("Experiments parameters saved at: ", logfile_parameters)

    cp_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=logfile_checkpoints,
        name_prefix=logfile,
        verbose=2,
    )

    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=cp_callback)
    model.save(logfile_model)
    print(f"Saved model at: {logfile_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    args = parser.parse_args()

    config_path = args.config
    main(config_path)
