from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


class QRExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, cnn_output_dim=256, direction_output_dim=64):
        super().__init__(observation_space, features_dim=cnn_output_dim + direction_output_dim)

        c, h, w = observation_space["image"].shape
        n_channels = c

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        sample = torch.zeros(1, n_channels, h, w)
        cnn_flat = self.cnn(sample).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_flat, cnn_output_dim),
            nn.ReLU()
        )

        self.qr_net  = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, direction_output_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        image = observations["image"].float() / 255.0
        qr_pos = observations["qr_pos"].float()

        cnn_features = self.cnn_linear(self.cnn(image))
        qr_features = self.qr_net(qr_pos)

        return torch.cat([cnn_features, qr_features], dim=1)
