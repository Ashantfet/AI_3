# slimevolleygym/algorithm/alphabeta.py

import numpy as np

ACTION_SPACE = [
    np.array([0, 0, 0]), np.array([1, 0, 0]),
    np.array([0, 1, 0]), np.array([0, 0, 1]),
    np.array([1, 1, 0]), np.array([0, 1, 1]),
    np.array([1, 0, 1])
]

def evaluation_function(obs):
    ball_x = obs[4]
    player_x = obs[0]
    return -abs(ball_x - player_x)

def alphabeta(env, obs, depth, alpha, beta, maximizing_player):
    if depth == 0:
        return evaluation_function(obs), None

    best_value = float("-inf") if maximizing_player else float("inf")
    best_action = None

    for action in ACTION_SPACE:
        state = env.clone_state()
        action_opponent = np.array([0, 0, 0])  # assume opponent does nothing

        if maximizing_player:
            joint_action = np.hstack([action, action_opponent])  # combine both actions
            new_obs, reward, done, info = env.step(joint_action)  # pass joint action
        else:
            joint_action = np.hstack([action, action_opponent])  # combine both actions
            new_obs, reward, done, info = env.step(joint_action)  # pass joint action

        value, _ = alphabeta(env, new_obs, depth - 1, alpha, beta, not maximizing_player)
        env.restore_state(state)

        if maximizing_player:
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
        else:
            if value < best_value:
                best_value = value
                best_action = action
            beta = min(beta, best_value)

        if beta <= alpha:
            break

    return best_value, best_action
