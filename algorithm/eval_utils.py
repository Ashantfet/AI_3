# your_algo/eval_utils.py
def evaluate_state(obs, game_over, info):
    # Score diff
    agent_score = info['agentScore']
    opponent_score = info['opponentScore']
    score = (agent_score - opponent_score) * 1000

    # Ball position bonus
    ball_x = obs[4]
    score += 100 * (ball_x - 0.5)  # encourage pushing ball right

    # Encourage moving toward ball
    agent_x, ball_x = obs[0], obs[4]
    score -= 10 * abs(agent_x - ball_x)

    # Game Over Bonus
    if game_over:
        if agent_score > opponent_score:
            score += 10000
        elif agent_score < opponent_score:
            score -= 10000

    return score
