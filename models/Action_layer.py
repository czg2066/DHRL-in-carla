import torch
import torch.nn as nn
from torch.distributions import Categorical

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_vehicles, features_per_vehicle)
        weights = self.attention_weights(x)  # (batch_size, num_vehicles, 1)
        weights = torch.softmax(weights, dim=1)  # Apply softmax to normalize the weights
        weighted_features = x * weights  # Element-wise multiplication
        aggregated_features = torch.sum(weighted_features, dim=1)  # Sum across vehicles
        return aggregated_features, weights

class CustomMLP(nn.Module):
    def __init__(self, Add_attention=True):
        if Add_attention:
            self.attention_layer = AttentionLayer(input_dim=4, attention_dim=50)  # assuming each vehicle has 4 features
        else: self.attention_layer = None
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.value_head = nn.Linear(64, 1)  # Value head

    def forward(self, x, action=None, debug=True):
        if self.attention_layer is not None: 
            x, weights = self.attention_layer(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        value = self.value_head(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action) # Log probability of the action
        if debug: print("Attention weights:", weights)
        return action, log_prob, value

# # Initialize the model
# mlp_model = MLP()

# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

# # Print the model summary
# print(mlp_model)
