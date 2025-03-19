# tennis-competition

Current projects trains a two DDPG agents to solve tennis competition unity environment.
![alt text](tennis-agents-match.gif)

## Project Details

### The Environment

States are defined in a 8-dimensional space. Action space size is 2-dimensional an continuous. The two dimensions refer to movement towards or away from the net and jumping.
The two DDPG agents are trained in an adversarial setting. The reward of each episode is taken from the maximum reward among both agents. Each agent has isolated views of the space state and take their own actions (independent networks) with their own replay buffers.

### The task

The task is episodic, meaning that it has a defined end state (marked by done flag coming from environment). An eposide is considered terminated when an agent lets the ball hit the ground OR kicks it out of bounds. The task is considered to be solved when one of the agents achieves an average reward of +0.5 over 100 consecutive episodes.

## Getting started

. Fork the `tennis-competition` repo on GitHub.

1. Clone your fork locally:

```bash
cd <directory_in_which_repo_should_be_created>
git clone https://github.com/julionevadod/tennis-competition.git
```

2. Now we need to install the environment. Navigate into the directory

```bash
cd tennis-competition
```

3. Then, install the environment with:

```bash
uv sync
```

4. Place Tennis.app from course resources inside **env** folder.

## Instructions

Once environment has been activated, training can be run from command line:

```bash
uv run train.py
```

Running the module as it is will run the training loop with default parameters. These default parameters produce an agent that solves the environment. However, agent hyperparameters can be configured by means of runtime arguments:

#### `-i`, `--iterations`

- **Description**: Number of environment steps to train for.
- **Default**: Value from `config["DEFAULT"]["ITERATIONS"]`.

#### `-b`, `--batch_size`

- **Description**: Batch size for learning steps.
- **Default**: Value from `config["DEFAULT"]["BATCH_SIZE"]`.

#### `-g`, `--gamma`

- **Description**: Discount rate used in reinforcement learning.
- **Default**: Value from `config["DEFAULT"]["GAMMA"]`.

#### `-a`, `--learning_rate_actor`

- **Description**: Learning rate for the Actor optimizer.
- **Default**: Value from `config["DEFAULT"]["LEARNING_RATE_ACTOR"]`.

#### `-c`, `--learning_rate_critic`

- **Description**: Learning rate for the Critic optimizer.
- **Default**: Value from `config["DEFAULT"]["LEARNING_RATE_CRITIC"]`.
