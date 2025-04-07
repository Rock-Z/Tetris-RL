import numpy as np
import gymnasium as gym
import torch
import cv2
from dqn_agent import DQNAgent

def preprocess_state(state):
    """Convert dictionary state into a multi-channel tensor."""
    # Extract components
    board = state['board'].astype(np.float32)
    active_mask = state['active_tetromino_mask'].astype(np.float32)
    holder = state['holder'].astype(np.float32)
    queue = state['queue'].astype(np.float32)
    
    # Normalize board values
    board = board / 8.0  # Assuming max value is 8
    
    # Stack components into channels
    # Shape: (channels, height, width)
    processed_state = np.stack([
        board,  # Main board state
        active_mask,  # Active tetromino position
        np.pad(holder, ((0,20), (0,14))),  # Pad holder to match board size
        np.pad(queue, ((0,20), (0,2)))  # Pad queue to match board size
    ], axis=0)
    
    return processed_state  # Shape will be (4, 24, 18)

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
    
    input_shape = processed_state.shape[1:]
    num_actions = env.action_space.n
    
    print(f"State shape: {input_shape}, Number of actions: {num_actions}")
    
    # Create and load the agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(input_shape, num_actions, device)
    
    # Load the trained model
    model_path = "dqn_model_final.pth"  # Update with your model path
    agent.load(model_path)
    
    # Set epsilon to a small value for some exploration during testing
    agent.epsilon = 0.05
    
    # Test the agent
    scores = test_agent(env, agent)
    
    print("Testing completed!")
