# Training an end-to-end Tetris Agent with DDQN

## Getting started

### Trying it out

Once you clone/download the repository, run the following to replicate training. Note that `uv` handles dependency installation, so you don't need to install any dependencies manually. If you do not have `uv`, consult [this guide](https://docs.astral.sh/uv/getting-started/installation/) or run `wget -qO- https://astral.sh/uv/install.sh | sh` if you're on a Unix system.

```bash
uv run train_dqn.py
```

Training takes significant time (on a server-grade GPU, this ranges from couple of hours to 1 day), so if you just want to try the model out, you can run the following to load a pre-trained model and play a game of Tetris:

```bash
uv run test_dqn.py --gif
```

Using `--gif` will save a gameplay by the agent to `gifs/test_episode.gif`. Or, you can use `--ansi` option to print out gameplays and evaluation statistics to the terminal. Note that the evaluation script uses the default reward function, which is different from the one used in training, so the scores may not be comparable. The model I trained is uploaded in the `final_checkpoints` directory, and is used as the default model checkpoint to evaluate in the test script.

## Env & Setup

- end-to-end: unlike common RL tetris agents, no action grouping/feature engineering.
- full-fledged tetris: agent sees pieces, has full action space, has hold
- vector envs: multiple parallel environments to speed up training
- logging: I implemented a minimal logger that saves key training stats as a csv file, which is later used for visualization

## Model

- DDQN: has a target net, used because DQN does not converge
    - 2 hidden layers, each of 128, ReLU activation

## Important Hyperparameters

Due to the computational cost, I did not do a full-fledged hyperparameter search (which would require running full-length experiments). Instead, I heuristically tried the below combinations and settled on the ones that seems to work best during early training.

- Reward function: the reward function turns out the be the most important factor: balancing reward values for different outcomes encourages drastically different behaviors. 
    - Game over: -1
    - Line clear: 1
    - Piece drop: 0.05
    - Staying alive: 0.05
- Model size
    - 2 hidden layers, each of 128, ReLU activation
- Learning rate
    - `2e-3`, after trying `1e-4`, `5e-3`, and `1e-2`
- Batch size
    - `256`, after trying `32`, `128`, and `512`
- Epsilon decay
    - `0.9999`, after trying `0.999`, `0.99999`, and `0.999999`. Epsilon starts at `1.0` and decays to `0.05` over the course of training.
- $\tau$ for target network update
    - `0.005`
- Update frequency
    - `2`, after trying `1`, `2`, and `4` 