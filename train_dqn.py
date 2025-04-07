import numpy as np
import gymnasium as gym
import torch
import cv2
from dqn_agent import DQNAgent

from tetris_gymnasium.envs.tetris import Tetris

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def train_dqn(env, agent, num_episodes=1000, max_steps=10000):
    scores = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        score = 0
        
        for step in range(max_steps):
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = preprocess_state(next_state)
            score += reward
            
            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward, done)
            
            # Move to the next state
            state = next_state
            
            # Perform one step of optimization
            loss = agent.train()
            
            if done:
                break
        
        scores.append(score)
        
        # Print episode stats
        avg_score = np.mean(scores[-100:])
        print(f"Episode {episode+1}, Score: {score}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Save the model periodically
        if (episode + 1) % 100 == 0:
            agent.save(f"dqn_model_episode_{episode+1}.pth")
    
    # Save the final model
    agent.save("dqn_model_final.pth")
    
    return scores

if __name__ == "__main__":
    # Create the environment
    env = gym.make("tetris_gymnasium/Tetris")
    
    # Get environment dimensions
    state, _ = env.reset()
    processed_state = preprocess_state(state)
    input_shape = processed_state.shape  # (channels, height, width)
    num_actions = env.action_space.n
    
    print(f"State shape: {input_shape}, Number of actions: {num_actions}")
    
    # Create the agent
    agent = DQNAgent(input_shape, num_actions, device)
    
    # Train the agent
    scores = train_dqn(env, agent)
    
    print("Training completed!")
