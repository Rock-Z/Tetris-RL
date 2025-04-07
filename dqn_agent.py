import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the flattened conv output
        conv_output_size = self._get_conv_output(input_shape)
        
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            x = F.relu(self.conv1(input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.shape))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done):
        # Ensure states are numpy arrays
        state = np.array(state)
        next_state = np.array(next_state)
        self.memory.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, input_shape, num_actions, device='cpu'):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        
        # Networks
        self.policy_net = DQN(input_shape, num_actions).to(device)
        self.target_net = DQN(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.update_target_every = 10
        self.batch_size = 32
        
        # Replay memory
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Training step counter
        self.training_step = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            # Convert state to tensor and add batch dimension
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Shape: (1, 4, 24, 18)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
            
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Extract and stack batch components properly
        state_batch = torch.FloatTensor(np.stack(batch[0])).to(self.device)  # Shape: (batch, channels, height, width)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.stack(batch[2])).to(self.device)
        reward_batch = torch.FloatTensor(batch[3]).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            
        # Compute target Q values
        target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
    
    def save(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
