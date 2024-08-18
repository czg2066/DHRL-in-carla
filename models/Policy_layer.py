import torch
import torch.nn as nn
from torchvision import models
from torch.distributions import Categorical

class CustomEfficientNet(nn.Module):
    def __init__(self):
        super(CustomEfficientNet, self).__init__()
        # Load pre-trained EfficientNetB0, exclude top layer
        self.base_model = models.efficientnet_b0(pretrained=True)
        self.base_model.features[0][0] = nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Add custom layers
        self.pooling = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc = nn.Linear(1280, 2)  # Assuming the output of the base model is 1280 features
        self.value_head = nn.Linear(1280, 1)  # Value head

    def forward(self, x, action=None):
        x = self.base_model.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        value = self.value_head(x) # Value head
        dist = Categorical(logits=logits) # Categorical distribution for action selection
        if action is None:
            action = dist.sample() # Sample an action from the distribution
        log_prob = dist.log_prob(action) # Log probability of the action
        return action, log_prob, value

# # Initialize the model
# model = CustomEfficientNet()

# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Model summary
# print(model)
