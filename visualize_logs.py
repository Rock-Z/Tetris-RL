import matplotlib.pyplot as plt
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Tetris RL training statistics')
    parser.add_argument('--csv', type=str, 
                        default="./logs/training_stats.csv",
                        help='Path to the CSV file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output image')
    parser.add_argument('--smooth', type=int, default=0,
                        help='Window size for moving average smoothing (0 for no smoothing)')
    return parser.parse_args()

def load_data(filepath):
    return pd.read_csv(filepath)

def smooth_data(data, window_size):
    if window_size <= 1:
        return data
    
    smoothed_data = data.copy()
    for col in ['Average_Score', 'Average_Lines', 'Epsilon', 'Loss']:
        smoothed_data[col] = data[col].rolling(window=window_size, center=True).mean()
    
    # Fill NaN values at the edges
    smoothed_data = smoothed_data.fillna(data)
    
    return smoothed_data

def visualize_training_stats(data, save_path=None, smooth_window=0):
    # Apply smoothing if requested
    if smooth_window > 1:
        plot_data = smooth_data(data, smooth_window)
        smooth_label = f" (Smoothed, window={smooth_window})"
    else:
        plot_data = data
        smooth_label = ""
    
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Tetris RL Training Statistics{smooth_label}', fontsize=16)
    
    # Plot Average Score
    axs[0].plot(plot_data['Episode'], plot_data['Average_Score'], color='#A7C7E7')  # Pastel blue
    max_score = data['Average_Score'].max()
    min_score = data['Average_Score'].min()
    last_score = data['Average_Score'].iloc[-1]
    axs[0].set_title(f'Average Score (Max: {max_score:.2f}, Min: {min_score:.2f}, Last: {last_score:.2f})')
    axs[0].set_ylabel('Average Score')
    axs[0].grid(True)
    
    # Plot Average Lines
    axs[1].plot(plot_data['Episode'], plot_data['Average_Lines'], color='#B6E2A1')  # Pastel green
    max_lines = data['Average_Lines'].max()
    min_lines = data['Average_Lines'].min()
    last_lines = data['Average_Lines'].iloc[-1]
    axs[1].set_title(f'Average Lines Cleared (Max: {max_lines:.2f}, Min: {min_lines:.2f}, Last: {last_lines:.2f})')
    axs[1].set_ylabel('Average Lines Cleared')
    axs[1].grid(True)
    
    # Plot Loss
    axs[2].plot(plot_data['Episode'], plot_data['Loss'], color='#FFB7B2')  # Pastel pink
    max_loss = data['Loss'].max()
    min_loss = data['Loss'].min()
    last_loss = data['Loss'].iloc[-1]
    axs[2].set_title(f'Loss (Max: {max_loss:.6f}, Min: {min_loss:.6f}, Last: {last_loss:.6f})')
    axs[2].set_ylabel('Loss')
    axs[2].set_xlabel('Episode')
    axs[2].grid(True)
    
    # Use log scale for loss plot as values can vary widely
    axs[2].set_yscale('log')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Load data
    data = load_data(args.csv)
    
    # Visualize
    visualize_training_stats(data, args.output, args.smooth)
