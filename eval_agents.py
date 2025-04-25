import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import os
import argparse
import numpy as np
import gym
import slimevolleygym
from time import sleep

from stable_baselines3 import PPO
from slimevolleygym import BaselinePolicy
from slimevolleygym.mlp import makeSlimePolicy, makeSlimePolicyLite

from zoo.minimax.minimax_agent import MinimaxAgent
from zoo.alphabeta.alphabeta_agent import AlphaBetaAgent

np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)

# === Agent Wrappers ===

class PPOPolicy:
    def __init__(self, path): self.model = PPO.load(path)
    def predict(self, obs): return self.model.predict(obs, deterministic=True)[0]

class RandomPolicy:
    def __init__(self, _): self.action_space = gym.spaces.MultiBinary(3)
    def predict(self, obs): return self.action_space.sample()

class MinimaxPolicy:
    def __init__(self, _): self.agent = MinimaxAgent(depth=2); self.env = None
    def predict(self, obs): return self.agent.act(self.env, obs)

class AlphaBetaPolicy:
    def __init__(self, _): self.agent = AlphaBetaAgent(depth=2); self.env = None
    def predict(self, obs): return self.agent.act(self.env, obs)

# === Evaluation Logic ===
def rollout(env, policy0, policy1, render_mode=False, max_steps=1000):
    obs0 = env.reset()
    obs1 = obs0
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < max_steps:
        action0 = policy0.predict(obs0)
        action1 = policy1.predict(obs1)

        obs0, reward, done, info = env.step(action0, action1)
        obs1 = info['otherObs']
        total_reward += reward

        if render_mode:
            env.render()
            sleep(0.05)

        steps += 1

    return total_reward


def evaluate_agents(env, policy0, policy1, trials=1000, render=False, seed=721):
    results = []
    for i in range(trials):
        env.seed(seed + i)
        score = rollout(env, policy0, policy1, render)
        print(f"Trial #{i}: score = {score}")
        results.append(score)
    return results

# === Main ===

if __name__ == "__main__":
    AGENTS = {
        "baseline": lambda _: BaselinePolicy(),
        "ppo": PPOPolicy,
        "cma": makeSlimePolicy,
        "ga": makeSlimePolicyLite,
        "random": RandomPolicy,
        "minimax": MinimaxPolicy,
        "alphabeta": AlphaBetaPolicy
    }

    DEFAULT_PATHS = {
        "ppo": "zoo/ppo/best_model.zip",
        "cma": "zoo/cmaes/slimevolley.cma.64.96.best.json",
        "ga": "zoo/ga_sp/ga.json",
        "baseline": None,
        "random": None,
        "minimax": None,
        "alphabeta": None
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--left', choices=AGENTS.keys(), default="baseline")
    parser.add_argument('--right', choices=AGENTS.keys(), default="ga")
    parser.add_argument('--leftpath', type=str, default="")
    parser.add_argument('--rightpath', type=str, default="")
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--day', action='store_true')
    parser.add_argument('--pixel', action='store_true')
    parser.add_argument('--seed', type=int, default=721)
    parser.add_argument('--trials', type=int, default=1000)
    args = parser.parse_args()

    if args.day:
        slimevolleygym.setDayColors()
    if args.pixel:
        slimevolleygym.setPixelObsMode()

    env = gym.make("SlimeVolley-v0")
    env.seed(args.seed)

    left_path = args.leftpath if args.leftpath else DEFAULT_PATHS[args.left]
    right_path = args.rightpath if args.rightpath else DEFAULT_PATHS[args.right]

    if args.leftpath and not os.path.exists(args.leftpath):
        raise FileNotFoundError(f"Left path {args.leftpath} does not exist.")
    if args.rightpath and not os.path.exists(args.rightpath):
        raise FileNotFoundError(f"Right path {args.rightpath} does not exist.")

    policy_left = AGENTS[args.left](left_path)
    policy_right = AGENTS[args.right](right_path)

    for agent, name in zip([policy_left, policy_right], [args.left, args.right]):
        if name in ["minimax", "alphabeta"]:
            agent.env = env

    scores = evaluate_agents(env, policy_right, policy_left, trials=args.trials, render=args.render)

    mean, std = np.mean(scores), np.std(scores)
    print("\nFinal Report:")
    print(f"{args.right} (Right) vs {args.left} (Left) over {args.trials} trials")
    print(f"Avg Score (Right - Left): {mean:.3f} ± {std:.3f}")
