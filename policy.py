import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os


class PolicyNet(nn.Module):
    def __init__(self, lr, num_actions, input_dims, fc1_dims, fc2_dims,
                 name, checkpoint_dir):
        super(PolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.output_layer = nn.Linear(fc2_dims, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f'----T.cuda.is_available(): {T.cuda.is_available()}')

    def forward(self, input_data):
        input_data = T.tensor(input_data, dtype=T.float32).to(self.device)

        layer1 = F.relu(self.fc1(input_data))
        layer2 = F.relu(self.fc2(layer1))
        outputs = self.output_layer(layer2)
        action_probs = F.softmax(outputs)

        return action_probs
    
    def save(self):
        print("----Saving checkpoint---")
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load(self):
        print("---Loading checkpoint")
        state_dict = T.load(self.checkpoint_file)
        self.load_state_dict(state_dict)