from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class DroneExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, cnn_output_dim=256, direction_output_dim=64):
        super().__init__(observation_space, features_dim=cnn_output_dim + direction_output_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        sample = torch.zeros(1, 1, 128, 128)
        cnn_flat = self.cnn(sample).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_flat, cnn_output_dim),
            nn.ReLU()
        )

        self.direction_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, direction_output_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        image = observations["image"].float() / 255.0
        direction = observations["direction"].float()

        cnn_features = self.cnn_linear(self.cnn(image))
        direction_features = self.direction_net(direction)

        return torch.cat([cnn_features, direction_features], dim=1)