import argparse
import configparser

import numpy as np
import torch
from src.agent import Agent
from unityagents import UnityEnvironment


def play(agent):
    env = UnityEnvironment(file_name="env/Tennis.app")
    brain_name = env.brain_names[0]
    num_agents = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)
    while True:
        with torch.no_grad():
            actions = np.vstack([
                agent.actor_local(torch.tensor(
                    states[0], dtype=torch.float32).to(device)).cpu().numpy(),
                agent.actor_local(torch.tensor(
                    states[1], dtype=torch.float32).to(device)).cpu().numpy(),
            ])
        actions = np.clip(actions, -1, 1)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break
    print(f"Score (max over agents) from episode: {np.max(scores)}")
    env.close()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("conf/conf.ini")

    parser = argparse.ArgumentParser(
        prog="Reacher Unity Environment: DDPG (Pseudo Actor-Critic) solution",
        description=""""
        Current program aims to solve Reacher Unity environment
        using policy based methods in the field of reinforcement learning
        """,
    )

    parser.add_argument(
        "-i", "--iterations", default=config["DEFAULT"]["ITERATIONS"], help="Number of environment steps to train for."
    )

    parser.add_argument(
        "-b", "--batch_size", default=config["DEFAULT"]["BATCH_SIZE"], help="Batch size for learning steps."
    )

    parser.add_argument(
        "-g", "--gamma", default=config["DEFAULT"]["GAMMA"], help="Discount Rate.")

    parser.add_argument(
        "-a",
        "--learning_rate_actor",
        default=config["DEFAULT"]["LEARNING_RATE_ACTOR"],
        help="Learning Rate for the Actor Optimizer.",
    )

    parser.add_argument(
        "-c",
        "--learning_rate_critic",
        default=config["DEFAULT"]["LEARNING_RATE_CRITIC"],
        help="Learning Rate for the Critic Optimizer.",
    )

    parser.add_argument(
        "-s",
        "--save_path",
        default=config["DEFAULT"]["SAVE_PATH"],
        help="Path to save checkpoints of the Agent networks.",
    )

    args = parser.parse_args()

    try:
        iterations = int(args.iterations)
    except ValueError as e:
        raise ValueError(
            f"(i)terations argument should be int. Provided value ({args.iterations}) cannot be casted."  # noqa: TRY003
        ) from e
    if iterations <= 0:
        raise ValueError(f"(i)terations argument should be a positive integer. Got {iterations}")  # noqa: TRY003

    try:
        batch_size = int(args.batch_size)
    except ValueError as e:
        raise ValueError(
            f"(b)atch_size argument should be int. Provided value ({args.batch_size}) cannot be casted."  # noqa: TRY003
        ) from e
    if batch_size <= 0:
        raise ValueError(f"(b)atch_size argument should be a positive integer. Got {batch_size}")  # noqa: TRY003

    try:
        gamma = float(args.gamma)
    except ValueError as e:
        raise ValueError(
            f"(g)amma argument should be a float number between 0 and 1. Provided value ({args.gamma}) cannot be casted."  # noqa: TRY003
        ) from e
    if (gamma < 0) or (gamma > 1):
        raise ValueError(f"(g)amma argument should be a float number between 0 and 1. Got {gamma}")  # noqa: TRY003

    try:
        lr_a = float(args.learning_rate_actor)
    except ValueError as e:
        raise ValueError(
            f"learning_rate_actor (a) argument should be a float number between 0 and 1. Provided value ({args.learning_rate_actor}) cannot be casted."  # noqa: TRY003
        ) from e
    if (lr_a < 0) or (lr_a > 1):
        raise ValueError(f"learning_rate_actor (a) argument should be a float number between 0 and 1. Got {lr_a}")  # noqa: TRY003

    try:
        lr_c = float(args.learning_rate_critic)
    except ValueError as e:
        raise ValueError(
            f"learning_rate_critic (c) argument should be a float number between 0 and 1. Provided value ({args.learning_rate_critic}) cannot be casted."  # noqa: TRY003
        ) from e
    if (lr_c < 0) or (lr_c > 1):
        raise ValueError(f"learning_rate_critic (c) argument should be a float number between 0 and 1. Got {lr_c}")  # noqa: TRY003

    save_path = args.save_path  # TODO: Proper exception handling of save path

    agent = Agent(gamma=gamma, lr_actor=lr_a, lr_critic=lr_c,
                  fc1_size=32, fc2_size=32, save_path=save_path)
    agent.load()
    play(agent)
