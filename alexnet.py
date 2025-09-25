import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---- 1. Device selection – works on CUDA, MPS (Apple Silicon) or CPU ---- 
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using {device} device")

# ---- 2. AlexNet model definition ---- 
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # Feature extractor – convolutional part
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Classifier – fully connected part
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),   # 6x6 is the size after conv/pool
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)   # flatten all dims except batch
        logits = self.classifier(x)
        return logits

# ---- 3. Example usage (for quick sanity check) ---- 
if __name__ == "__main__":
    # Simple test on random input
    model = AlexNet(num_classes=1000).to(device)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    output = model(dummy_input)
    print("Output shape:", output.shape)   # Should be [1, 1000]