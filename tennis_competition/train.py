import argparse
import collections
import configparser

import numpy as np
import torch
from src.agent import Agent
from src.OUNoise import OUNoise
from src.replay_buffer import ExperienceReplayBuffer
from unityagents import UnityEnvironment

MAX_LEN_EPISODE = 1000000
WINDOW_SIZE = 100
BUFFER_SIZE = 100000


def train(n_iterations, agent, batch_size, update_every, sigma, sigma_decay):
    env = UnityEnvironment(file_name="env/Tennis.app")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    replay_buffer = ExperienceReplayBuffer(BUFFER_SIZE)

    noise = OUNoise(action_size, 0, sigma=sigma, sigma_decay=sigma_decay)

    scores = []
    scores_window = collections.deque(maxlen=WINDOW_SIZE)
    brain_name = env.brain_names[0]

    for i in range(1, n_iterations + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(2)
        for j in range(1, MAX_LEN_EPISODE + 1):
            with torch.no_grad():
                actions = np.vstack([
                    agent.actor_local(torch.tensor(states[0], dtype=torch.float32).to(device)).cpu().numpy()
                    + noise.sample(),
                    agent.actor_local(torch.tensor(states[1], dtype=torch.float32).to(device)).cpu().numpy()
                    + noise.sample(),
                ])
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += rewards
            experiences = list(
                zip(
                    np.split(states, states.shape[0]),
                    np.split(actions, actions.shape[0]),
                    np.split(next_states, next_states.shape[0]),
                    rewards,
                    dones,
                )
            )

            replay_buffer.insert(experiences[0])
            replay_buffer.insert(experiences[1])

            if (j % update_every == 0) & (len(replay_buffer) >= batch_size):
                experiences = replay_buffer.sample(batch_size)
                agent.update(experiences)

            if any(dones):
                break

            states = next_states

        scores.append(score.max())
        scores_window.append(score.max())

        print(
            f"\rITERATION {i}/{n_iterations}: Average Reward Last 100: {float(np.mean(scores_window)):.2f} \t Last Episode: {max(score):.2f}",
            end="",
        )

        if float(np.mean(scores_window) > 0.5):
            print(f"\nEnvironment solved in {i} iterations with a score of {float(np.mean(scores_window)):.2f}")
            agent.save()
            break

        if i % 100 == 0:
            noise.reset()
            print(
                f"\rITERATION {i}/{n_iterations}: Average Reward Last 100: {float(np.mean(scores_window)):.2f} \t Last Episode: {max(score):.2f}"
            )

    env.close()
    return scores


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

    parser.add_argument("-g", "--gamma", default=config["DEFAULT"]["GAMMA"], help="Discount Rate.")

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
        "-u", "--update_every", default=config["DEFAULT"]["UPDATE_EVERY"], help="Number of steps between updates."
    )

    parser.add_argument(
        "-s",
        "--sigma",
        default=config["DEFAULT"]["SIGMA"],
        help="Standard deviation for normal noise.",
    )

    parser.add_argument(
        "-d",
        "--sigma_decay",
        default=config["DEFAULT"]["SIGMA_DECAY"],
        help="Decay of normal noise.",
    )

    parser.add_argument(
        "-p",
        "--save_path",
        default=config["DEFAULT"]["SAVE_PATH"],
        help="Path to save checkpoints of the Agent networks.",
    )

    args = parser.parse_args()

    try:
        iterations = int(args.iterations)
    except ValueError as e:
        raise ValueError(
            f"(i)terations argument should be int. Provided value ({args.iterations}) cannot be casted."
        ) from e
    if iterations <= 0:
        raise ValueError(f"(i)terations argument should be a positive integer. Got {iterations}")  # noqa: TRY003

    try:
        batch_size = int(args.batch_size)
    except ValueError as e:
        raise ValueError(
            f"(b)atch_size argument should be int. Provided value ({args.batch_size}) cannot be casted."
        ) from e
    if batch_size <= 0:
        raise ValueError(f"(b)atch_size argument should be a positive integer. Got {batch_size}")  # noqa: TRY003

    try:
        gamma = float(args.gamma)
    except ValueError as e:
        raise ValueError(
            f"(g)amma argument should be a float number between 0 and 1. Provided value ({args.gamma}) cannot be casted."
        ) from e
    if (gamma < 0) or (gamma > 1):
        raise ValueError(f"(g)amma argument should be a float number between 0 and 1. Got {gamma}")  # noqa: TRY003

    try:
        lr_a = float(args.learning_rate_actor)
    except ValueError as e:
        raise ValueError(
            f"learning_rate_actor (a) argument should be a float number between 0 and 1. Provided value ({args.learning_rate_actor}) cannot be casted."
        ) from e
    if (lr_a < 0) or (lr_a > 1):
        raise ValueError(f"learning_rate_actor (a) argument should be a float number between 0 and 1. Got {lr_a}")  # noqa: TRY003

    try:
        lr_c = float(args.learning_rate_critic)
    except ValueError as e:
        raise ValueError(
            f"learning_rate_critic (c) argument should be a float number between 0 and 1. Provided value ({args.learning_rate_critic}) cannot be casted."
        ) from e
    if (lr_c < 0) or (lr_c > 1):
        raise ValueError(f"learning_rate_critic (c) argument should be a float number between 0 and 1. Got {lr_c}")  # noqa: TRY003

    try:
        update_every = int(args.update_every)
    except ValueError as e:
        raise ValueError(
            f"(u)pdate_every argument should be int. Provided value ({args.update_every}) cannot be casted."
        ) from e
    if update_every <= 0:
        raise ValueError(f"(u)pdate_every argument should be a positive integer. Got {update_every}")  # noqa: TRY003

    try:
        sigma = float(args.sigma)
    except ValueError as e:
        raise ValueError(
            f"(s)igma argument should be a float number between 0 and 1. Provided value ({args.sigma}) cannot be casted."
        ) from e
    if (sigma < 0) or (sigma > 1):
        raise ValueError(f"(s)igma argument should be a float number between 0 and 1. Got {sigma}")  # noqa: TRY003

    try:
        sigma_decay = float(args.sigma_decay)
    except ValueError as e:
        raise ValueError(
            f"sigma_(d)ecay argument should be a float number between 0 and 1. Provided value ({args.sigma_decay}) cannot be casted."
        ) from e
    if (sigma_decay < 0) or (sigma_decay > 1):
        raise ValueError(f"sigma_(d)ecay argument should be a float number between 0 and 1. Got {sigma_decay}")  # noqa: TRY003

    save_path = args.save_path  # TODO: Proper exception handling of save path

    agent = Agent(gamma=gamma, lr_actor=lr_a, lr_critic=lr_c, fc1_size=32, fc2_size=32, save_path=save_path)

    train(iterations, agent, batch_size, update_every, sigma, sigma_decay)
