import numpy as np
import gymnasium as gym
import torch
import imageio
import os
from dqn_agent import DQNAgent, ReplayMemory

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.rewards import RewardsMapping


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
    
    # Stack components into channels
    processed_state = np.stack([
        board,  # Main board state
        active_mask,  # Active tetromino position
        np.pad(holder, ((0,20), (0,14))),  # Pad holder to match board size
        np.pad(queue, ((0,20), (0,2)))  # Pad queue to match board size
    ], axis=0)
    
    return processed_state

def record_episode(env, agent, filename):
    """Record one episode and save it as a GIF."""
    frames = []
    state, _ = env.reset()
    state = preprocess_state(state)
    done = False
    
    while not done:
        # Get frame
        frame = env.render()
        frames.append(frame)
        
        # Select and perform action
        action = agent.select_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = preprocess_state(next_state)
    
    # Save as GIF
    imageio.mimsave(filename, frames, fps=10)

def train_dqn(env, agent, num_episodes=2000, max_steps=10000):
    # Update hyperparameters for better exploration
    agent.memory = ReplayMemory(1_000_000)
    agent.gamma = 0.99
    agent.epsilon = 1.0
    agent.epsilon_min = 0.05  
    agent.epsilon_decay = 1 - 0.05/num_episodes  
    agent.learning_rate = 1e-4 
    agent.update_target_every = 2
    agent.batch_size = 512
    agent.tau = 0.005  
    
    scores = []
    lines_cleared_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        score = 0
        total_lines_cleared = 0
        
        for step in range(max_steps):
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if not terminated:
                reward += 0.05 # alive bonus
            done = terminated or truncated
            
            # Track lines cleared
            total_lines_cleared += info.get('lines_cleared', 0)
            
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
        lines_cleared_history.append(total_lines_cleared)
        
        # Calculate averages
        avg_score = np.mean(scores[-100:])
        avg_lines = np.mean(lines_cleared_history[-100:])
        
        print(f"Episode {episode+1}, Score: {score}, Lines: {total_lines_cleared}, "
              f"Avg Score: {avg_score:.2f}, Avg Lines: {avg_lines:.2f}, "
              f"Epsilon: {agent.epsilon:.2f}")
        
        # Save the model and record episode periodically
        if (episode + 1) % 100 == 0:
            model_filename = f"dqn_model_episode_{episode+1}.pth"
            gif_filename = os.path.join("gifs", f"episode_{episode+1}.gif")
            
            # Save model
            agent.save(os.path.join("checkpoints", model_filename))
            
            # Record and save episode
            record_episode(env, agent, gif_filename)
            print(f"Saved model and episode recording: {gif_filename}")
    
    # Save final model and episode
    agent.save("checkpoints/dqn_model_final.pth")
    record_episode(env, agent, "gifs/final_episode.gif")
    
    return scores

if __name__ == "__main__":
    # Create the environment
    rewards_mapping = RewardsMapping()
    rewards_mapping.alife = 0
    rewards_mapping.game_over = -1
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array", rewards_mapping=rewards_mapping)
    
    # Get environment dimensions
    state, _ = env.reset()
    processed_state = preprocess_state(state)
    input_shape = processed_state.shape  # Should be (4, 24, 18)
    num_actions = env.action_space.n
    
    print(f"State shape: {input_shape}, Number of actions: {num_actions}")
    
    # Create the agent
    agent = DQNAgent(input_shape, num_actions, device)
    
    # Train the agent
    scores = train_dqn(env, agent)
    
    print("Training completed!")
