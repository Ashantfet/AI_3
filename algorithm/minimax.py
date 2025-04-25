# slimevolleygym/algorithm/minimax.py

import numpy as np

ACTION_SPACE = [
    np.array([0, 0, 0]),  # do nothing
    np.array([1, 0, 0]),  # left
    np.array([0, 1, 0]),  # jump
    np.array([0, 0, 1]),  # right
    np.array([1, 1, 0]),  # left + jump
    np.array([0, 1, 1]),  # right + jump
    np.array([1, 0, 1]),  # left + right (invalid, but harmless)
]

def evaluation_function(obs):
    # Simple evaluation: reward distance to ball & ball x position
    ball_x = obs[4]
    player_x = obs[0]
    return -abs(ball_x - player_x)  # closer to ball is better

def minimax(env, obs, depth, maximizing_player):
    if depth == 0:
        return evaluation_function(obs), None

    best_value = float("-inf") if maximizing_player else float("inf")
    best_action = None

    for action in ACTION_SPACE:
        state = env.clone_state()
        action_opponent = np.array([0, 0, 0])  # assume opponent does nothing

        if maximizing_player:
            joint_action = np.hstack([action, action_opponent])
            new_obs, reward, done, info = env.step(joint_action)  # single joint action
        else:
            joint_action = np.hstack([action, action_opponent])
            new_obs, reward, done, info = env.step(joint_action)  # single joint action

        value, _ = minimax(env, new_obs, depth - 1, not maximizing_player)
        env.restore_state(state)

        if maximizing_player and value > best_value:
            best_value = value
            best_action = action
        elif not maximizing_player and value < best_value:
            best_value = value
            best_action = action

    return best_value, best_action
