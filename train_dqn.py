import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
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
    
    # Extract components for single state
    board = state['board'].astype(np.float32)
    active_mask = state['active_tetromino_mask'].astype(np.float32)
    holder = state['holder'].astype(np.float32)
    queue = state['queue'].astype(np.float32)
    
    # Stack components into channels
    # Convert to tensors
    if len(state['board'].shape) == 2:
        # Handle single state
        board_tensor = torch.from_numpy(board)
        active_mask_tensor = torch.from_numpy(active_mask)
        holder_tensor = torch.from_numpy(np.pad(holder, ((0,20), (0,14))))
        queue_tensor = torch.from_numpy(np.pad(queue, ((0,20), (0,2))))
        
        processed_state = torch.stack([
            board_tensor,  # Main board state
            active_mask_tensor,  # Active tetromino position
            holder_tensor,  # Holder state
            queue_tensor  # Queue state
        ], dim=0)
    else:
        # Handle batch of states
        assert len(state['board'].shape) == 3
        board_tensor = torch.from_numpy(board)
        active_mask_tensor = torch.from_numpy(active_mask)
        holder_tensor = torch.from_numpy(np.pad(holder, ((0,0), (0,20), (0,14))))
        queue_tensor = torch.from_numpy(np.pad(queue, ((0,0), (0,20), (0,2))))
        
        processed_state = torch.stack([
            board_tensor,  # Main board state
            active_mask_tensor,  # Active tetromino position
            holder_tensor,  # Holder state
            queue_tensor  # Queue state
        ], dim=1)
        
    return processed_state

def make_env(idx):
    """Create a single environment instance with a unique seed."""
    def _init():
        rewards_mapping = RewardsMapping()
        rewards_mapping.alife = 0.05
        env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array", rewards_mapping=rewards_mapping)
        env.reset(seed=idx)
        return env
    return _init

def record_episode(envs, agent, filename):
    """Record one episode from the first environment and save it as a GIF."""
    # Create a separate environment for recording
    record_env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    frames = []
    state, _ = record_env.reset()
    state = preprocess_state(state)
    done = False
    
    while not done:
        # Select action
        action = agent.select_action(state.to(device), epsilon=0.0)
        if type(action) is np.ndarray:
            action = action[0] #patch fix, if action came as a batch
        
        # Execute action
        next_state, reward, terminated, truncated, _ = record_env.step(action)
        done = terminated or truncated
        
        # Store frame
        frame = record_env.render()
        frames.append(frame)
        
        # Update state
        state = preprocess_state(next_state)
    
    # Save as GIF
    imageio.mimsave(filename, frames, fps=10)

def train_dqn(envs, agent, num_envs=4, num_episodes=10000, max_steps=10000):
    # Update hyperparameters for better exploration
    agent.memory = ReplayMemory(100_000)
    agent.gamma = 0.99
    agent.epsilon = 1.0
    agent.epsilon_min = 0.05  
    agent.epsilon_decay = 0.9999
    agent.learning_rate = 2e-3
    agent.update_target_every = 2
    agent.batch_size = 256
    agent.tau = 0.005  
    
    scores = []
    lines_cleared_history = []
    
    # Track episodes per environment
    episode_counts = [0] * num_envs
    total_episodes = 0
    
    # Initialize states
    states, _ = envs.reset()
    states = preprocess_state(states)
    
    # Track scores and lines for current episodes
    current_scores = np.zeros(num_envs)
    current_lines_cleared = np.zeros(num_envs)
    
    while total_episodes < num_episodes:
        # Select and perform actions for all environments
        actions = agent.select_action(states.to(device))
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        
        # Apply alive bonus
        dones = terminateds | truncateds
        rewards = np.where(~terminateds, rewards + 0.05, rewards)  # alive bonus
        
        # Process next states
        next_states = preprocess_state(next_states)
        
        # Track score and lines per environment
        current_scores += rewards
        
        # Track lines cleared from info
        if 'lines_cleared' in infos:
            current_lines_cleared += infos['lines_cleared']
        
        # Store transitions in memory
        for i in range(num_envs):
            agent.memory.push(states[i], actions[i], next_states[i], rewards[i], dones[i])
        
        # Update states
        states = next_states
        
        # Handle completed episodes
        for i in range(num_envs):
            if dones[i]:
                # Log and record completed episode
                scores.append(current_scores[i])
                lines_cleared_history.append(current_lines_cleared[i])
                
                # Reset tracking for this environment
                current_scores[i] = 0
                current_lines_cleared[i] = 0
                episode_counts[i] += 1
                total_episodes += 1
                
                # Print progress
                if total_episodes % (num_episodes // 100) == 0:
                    avg_score = np.mean(scores[-100:]) if scores else 0
                    avg_lines = np.mean(lines_cleared_history[-100:]) if lines_cleared_history else 0
                    print(f"Episode {total_episodes}, Avg Score: {avg_score:.2f}, "
                          f"Avg Lines: {avg_lines:.2f}, Epsilon: {agent.epsilon:.2f}")
                
                # Save model and record periodically
                if total_episodes % (num_episodes // 10) == 0:
                    model_filename = f"dqn_model_episode_{total_episodes}.pth"
                    gif_filename = os.path.join("gifs", f"episode_{total_episodes}.gif")
                    
                    # Save model
                    agent.save(os.path.join("checkpoints", model_filename))
                    
                    # Record and save episode
                    record_episode(envs, agent, gif_filename)
                    print(f"Saved model and episode recording: {gif_filename}")
        
        # Perform optimization step
        loss = agent.train()
    
    # Save final model and episode
    agent.save("checkpoints/dqn_model_final.pth")
    record_episode(envs, agent, "gifs/final_episode.gif")
    
    return scores

if __name__ == "__main__":
    # Number of parallel environments
    num_envs = 8
    
    # Create vectorized environment
    env_fns = [make_env(i) for i in range(num_envs)]
    envs = AsyncVectorEnv(env_fns)
    
    # Get environment dimensions from a single env (for reference)
    single_env = gym.make("tetris_gymnasium/Tetris")
    state, _ = single_env.reset()
    processed_state = preprocess_state(state)
    input_shape = processed_state.shape  # Should be (4, 24, 18)
    num_actions = single_env.action_space.n
    single_env.close()
    
    print(f"State shape: {input_shape}, Number of actions: {num_actions}")
    print(f"Training with {num_envs} parallel environments")
    
    # Create the agent
    agent = DQNAgent(input_shape, num_actions, device)
    
    # Train the agent
    scores = train_dqn(envs, agent, num_envs=num_envs)
    
    # Clean up
    envs.close()
    print("Training completed!")
