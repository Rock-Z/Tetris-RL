import numpy as np
import gymnasium as gym
import torch
import cv2
from dqn_agent import DQNAgent
from train_dqn import preprocess_state

def test_agent(env, agent, num_episodes=10, render=True):
    scores = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        score = 0
        done = False
        
        while not done:
            if render:
                env_state = env.render()
                print(env_state)
                cv2.waitKey(100)  # Slow down visualization
            
            # Select action (no exploration)
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            score += reward
            state = preprocess_state(next_state)
        
        scores.append(score)
        print(f"Episode {episode+1}, Score: {score}")
    
    print(f"Average Score: {np.mean(scores)}")
    return scores

if __name__ == "__main__":
    # Create the environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    
    # Get environment dimensions
    state, _ = env.reset()
    
    # Process the state to get the right shape for the network
    processed_state = preprocess_state(state)
    num_actions = env.action_space.n
    
    print(f"State shape: {processed_state}, Number of actions: {num_actions}")
    
    # Create and load the agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(processed_state, num_actions, device)
    
    # Load the trained model
    model_path = "checkpoints/dqn_model_final.pth"
    agent.load(model_path)
    
    # Set epsilon to 0 for testing
    agent.epsilon = 0
    
    # Test the agent
    scores = test_agent(env, agent)
    
    print("Testing completed!")
