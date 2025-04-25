import gym
import time
import slimevolleygym
from slimevolleygym.patch_env import patch_env
from slimevolleygym.slimevolley import SlimeVolleyEnv
from algorithm.alphabeta import alphabeta
from algorithm.random_agent import RandomAgent

import numpy as np
import imageio
import cv2

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

def run_alphabeta_game():
    env = gym.make("SlimeVolley-v0")
    obs = env.reset()
    frames = []
    score_left = 0
    score_right = 0

    fps = 30
    duration_sec = 15
    total_frames = fps * duration_sec

    opponent = RandomAgent()

    start_time = time.time()

    for step in range(total_frames):
        frame = render_frame_with_score(env, score_left, score_right)
        frames.append(frame)

        _, action_left = alphabeta(env, obs, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)

        if action_left is None:
            action_left = np.array([0, 0, 0])

        action_right = opponent.act(obs)

        obs, reward, done, _ = env.step(action_left, action_right)

        if reward == 1:
            score_left += 1
        elif reward == -1:
            score_right += 1

        if done:
            obs = env.reset()

    final_frame = render_frame_with_score(env, score_left, score_right, final=True)
    for _ in range(fps * 2):
        frames.append(final_frame)

    env.close()

    height, width, _ = frames[0].shape
    height = ((height + 15) // 16) * 16
    width = ((width + 15) // 16) * 16
    resized_frames = [cv2.resize(f, (width, height)) for f in frames]

    out_path = "alphabeta_fullgame.mp4"
    imageio.mimsave(out_path, resized_frames, fps=fps)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Saved 15-second AlphaBeta game to {out_path}")
    print(f"Total frames played: {len(frames)}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print(f"Final Score: Yellow={score_left}, Blue={score_right}")

if __name__ == "__main__":
    run_alphabeta_game()
