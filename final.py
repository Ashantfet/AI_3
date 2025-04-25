import gym
import slimevolleygym
from slimevolleygym.patch_env import patch_env
from slimevolleygym.slimevolley import SlimeVolleyEnv
from algorithm.minimax import minimax
from algorithm.alphabeta import alphabeta
from algorithm.random_agent import RandomAgent

import numpy as np
import imageio
import cv2
import time

patch_env()

def render_frame_with_score(env, score_left, score_right, final=False):
    frame = env.render(mode="rgb_array")
    overlay = frame.copy()
    text_color = (255, 255, 255)

    cv2.putText(overlay, f"Yellow: {score_left} | Blue: {score_right}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

    if final:
        if score_left > score_right:
            winner = "Yellow Wins!"
        elif score_right > score_left:
            winner = "Blue Wins!"
        else:
            winner = "It's a Draw!"
        cv2.putText(overlay, winner, (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)

    return overlay

def run_game(algorithm="minimax"):
    env = gym.make("SlimeVolley-v0")
    obs = env.reset()
    frames = []
    score_left = 0
    score_right = 0

    fps = 30  # Original FPS (you can adjust for effect)
    duration_sec = 30  # For 0.5x speed, record for twice the original duration
    total_frames = fps * duration_sec

    opponent = RandomAgent()
    frame_count = 0
    start_time = time.time()

    for step in range(total_frames):
        frame = render_frame_with_score(env, score_left, score_right)
        frames.append(frame)

        if algorithm == "minimax":
            _, action_left = minimax(env, obs, depth=3, maximizing_player=True)
        elif algorithm == "alphabeta":
            _, action_left = alphabeta(env, obs, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)

        if action_left is None:
            action_left = np.array([0, 0, 0])

        action_right = opponent.act(obs)

        # Combine both actions into one joint action
        joint_action = np.hstack([action_left, action_right])
        obs, reward, done, _ = env.step(joint_action)  # pass joint action

        frame_count += 1

        if reward == 1:
            score_left += 1
        elif reward == -1:
            score_right += 1

        if done:
            obs = env.reset()

    end_time = time.time()
    duration = end_time - start_time

    # Append final frame for 2 seconds
    final_frame = render_frame_with_score(env, score_left, score_right, final=True)
    for _ in range(fps * 2):
        frames.append(final_frame)

    env.close()

    # Get dimensions for uniformity
    height, width, _ = frames[0].shape
    height = ((height + 15) // 16) * 16
    width = ((width + 15) // 16) * 16

    # Write frames with resize on-the-fly
    out_path = f"{algorithm}_fullgame.mp4"
    with imageio.get_writer(out_path, fps=fps ) as writer:  # Write at 0.5x speed (half FPS)
        for f in frames:
            resized = cv2.resize(f, (width, height))
            writer.append_data(resized)

    print(f"✅ Saved full 30-second game  {out_path}")
    print(f"\n📊 Match Stats ({algorithm.capitalize()}):")
    print(f"Total Score: Yellow {score_left} - Blue {score_right}")
    print(f"Total Frames Played: {frame_count}")
    print(f"Execution Time: {duration:.2f} seconds")

def run_both_algorithms():
    # Run Minimax first
    print("Running Minimax algorithm...")
    run_game(algorithm="minimax")

    # Run Alpha-Beta after Minimax
    print("\nRunning Alpha-Beta algorithm...")
    run_game(algorithm="alphabeta")

if __name__ == "__main__":
    run_both_algorithms()
