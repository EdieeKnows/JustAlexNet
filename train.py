import torch
import torch.nn as nn
from alexnet import AlexNet
from dataset_imagenett import train_loader, val_loader

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    # ---- 1. Device selection – works on CUDA, MPS (Apple Silicon) or CPU ---- 
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using {device} device")
    # Define the model and optimizer
    model = AlexNet(num_classes=1000).to(device)
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    # Initialize the optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,                     # initial learning rate
        momentum=0.9,                # classic momentum
        weight_decay=5e-4,           # L2 regularisation
        nesterov=True                # optional
    )

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, batch_size=32)
    print("Done!")


