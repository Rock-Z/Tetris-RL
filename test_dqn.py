import numpy as np
import gymnasium as gym
import torch
import argparse
from dqn_agent import DQNAgent
from train_dqn import preprocess_state, record_episode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_agent(env, agent, num_episodes=10, render=True):
    scores = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        score = 0
        done = False
        step_ct = 0
        
        while not done:
            if render:
                env_state = env.render()
                # Clear previous output and print new state
                print("Episode:", episode + 1, "Score:", score, "Step:", step_ct)
                print(env_state)
            
            # Select action (no exploration)
            action = agent.select_action(state.to(device), epsilon=0)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action[0])
            step_ct += 1
            done = terminated or truncated
            
            score += reward
            state = preprocess_state(next_state)
        
        scores.append(score)
        print(f"Episode {episode+1}, Score: {score}")
    
    print(f"Average Score: {np.mean(scores)}")
    return scores

if __name__ == "__main__":
    # script has 2 options: ansi print or save gif
    parser = argparse.ArgumentParser(description="Test a DQN agent on Tetris.")
    parser.add_argument("--gif", action="store_true", help="Record a GIF of the agent playing.")
    parser.add_argument("--ansi", action="store_true", help="Print the game state in ANSI format.")
    parser.add_argument("--model", type=str, default="final_checkpoints/dqn_model_final.pth", help="Path to the model file.")
    args = parser.parse_args()
    if args.gif and args.ansi:
        raise ValueError("Choose either --gif or --ansi, not both.")

    # Create the environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    
    # Get environment dimensions
    state, _ = env.reset()
    
    # Process the state to get the right shape for the network
    processed_state = preprocess_state(state)
    num_actions = env.action_space.n
    
    print(f"State shape: {processed_state.shape}, Number of actions: {num_actions}")
    
    # Create and load the agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(processed_state.shape, num_actions, device)
    
    # Load the trained model
    model_path = "checkpoints/dqn_model_final.pth"
    agent.load(model_path)
    agent.policy_net.to(device)
    agent.target_net.to(device)
    
    # Set epsilon to 0 for testing
    agent.epsilon = 0
    
    # Test the agent
    if args.gif:
        gif_filename = "gifs/test_episode.gif"
        record_episode(agent, gif_filename)
    elif args.ansi:
        print("Testing in ANSI mode...")
        scores = test_agent(env, agent, render=True)
    
    print("Testing completed!")
