import torch
from torch import nn

# Model script. Add description later. 

class Model(nn.Module):
    def __init__(self, model_channels: int = 16, multiplier: list = [1,2,3]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, model_channels * multiplier[0], 3, 1)
        self.conv2 = nn.Conv2d(model_channels * multiplier[0], model_channels * multiplier[1], 3, 1)
        self.conv3 = nn.Conv2d(model_channels * multiplier[1], model_channels * multiplier[2], 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(48 * 46 * 46, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        output = self.fc1(x).view(-1)
        return output


if __name__ == "__main__":
    model = Model()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 384, 384)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
