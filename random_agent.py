import cv2
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env.reset(seed=42)

    terminated = False
    while not terminated:
        state = env.render()
        print(state)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        key = cv2.waitKey(100) # timeout to see the movement
    print("Game Over!")
